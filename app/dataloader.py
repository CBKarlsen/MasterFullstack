# dataloader.py
"""
Reads sensor recordings (CSV/Excel) and turns them into a stream of samples.

A single :class:`DataStreamer` can point at one file or a whole folder of files.
For each file it:
  * loads everything as strings first, so messy real-world data never crashes
    the parser (European decimal commas, stray text, ghost columns, etc.);
  * auto-detects the sampling rate from the timestamp column (defaults to 20 Hz
    when no usable time column exists);
  * exposes every numeric column for the UI to plot, while also mapping the few
    "known" columns (flow, inlet/outlet pressure) to fixed keys for the detector.

``stream()`` yields one dict per row, with a monotonically increasing ``time``
that continues across files in a folder so multi-file recordings stitch into a
single continuous timeline.
"""

import pandas as pd
import os
import numpy as np
from typing import Dict, List, Any, Optional, Generator


class ColumnMetadata:
    """Metadata about a column in the data file."""

    def __init__(self, name: str, col_type: str = "numeric", unit: str = ""):
        self.name = name
        self.col_type = col_type
        self.unit = unit

    def to_dict(self) -> Dict[str, str]:
        return {"name": self.name, "type": self.col_type, "unit": self.unit}


class DataStreamer:
    """
    Acts as the 'Source'.
    - Handles mixed file types (.csv, .xlsx).
    - Auto-detects column names.
    - Auto-detects Sampling Rate (Hz) from Time column.
    - Extracts ALL numeric columns for user selection.
    """

    def __init__(self, path: str):
        self.path = path
        self.columns: List[ColumnMetadata] = []
        self._column_names: List[str] = []

    @staticmethod
    def get_file_columns(filepath: str) -> List[Dict[str, str]]:
        """
        Get metadata about all columns in a data file.

        Args:
            filepath: Path to CSV or Excel file.

        Returns:
            List of column metadata dicts with name, type, and unit.
        """
        try:
            if filepath.endswith(".csv"):
                df = pd.read_csv(
                    filepath, sep=None, dtype=str, engine="python", nrows=10
                )
            elif filepath.endswith(".xlsx"):
                df = pd.read_excel(filepath, engine="calamine", nrows=10)
            else:
                return []
        except Exception as e:
            print(f"Error reading file for columns: {e}")
            return []

        columns = []
        # Known unit mappings
        unit_hints = {
            "pressure": "bar",
            "flow": "L/s",
            "temperature": "°C",
            "temp": "°C",
            "time": "s",
        }

        for col in df.columns:
            col_lower = col.lower()
            unit = ""
            for hint, u in unit_hints.items():
                if hint in col_lower:
                    unit = u
                    break

            columns.append({"name": col, "type": "numeric", "unit": unit})

        return columns

    def stream(self) -> Generator[Dict[str, Any], None, None]:
        """Yield one sample dict per row across all resolved files.

        Each yielded dict has: ``time`` (seconds, continuous across files),
        ``flow``/``p_in``/``p_out`` (mapped known columns, 0.0 if absent),
        ``raw`` (every numeric column for this row), and ``columns`` (the list
        of numeric column names). Unreadable files are skipped, not fatal.
        """
        files_to_read = []

        # 1. Resolve Files
        if os.path.isdir(self.path):
            all_files = sorted(os.listdir(self.path))
            for f in all_files:
                if f.endswith(".csv") or f.endswith(".xlsx"):
                    files_to_read.append(os.path.join(self.path, f))
            print(f"Pipeline: Queued {len(files_to_read)} files.")
        elif os.path.isfile(self.path):
            files_to_read.append(self.path)
        else:
            raise ValueError("Path not found.")

        global_time_counter = 0.0

        # 2. Iterate Files
        for file_path in files_to_read:
            print(f"Streaming: {file_path}")

            # LOAD DATA
            try:
                if file_path.endswith(".csv"):
                    df = pd.read_csv(file_path, sep=None, dtype=str, engine="python")
                elif file_path.endswith(".xlsx"):
                    df = pd.read_excel(file_path, engine="calamine")
                else:
                    continue
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
                continue

            try:
                # Parse one cell into a float: accepts European decimal commas
                # ("1,5" -> 1.5) and coerces anything unparseable or non-finite
                # (NaN/Inf, blank cells, text) to 0.0 so a bad cell can't crash
                # the stream.
                def to_float(x):
                    if isinstance(x, (int, float)):
                        v = float(x)
                        return 0.0 if (np.isnan(v) or np.isinf(v)) else v
                    try:
                        v = float(str(x).replace(",", "."))
                        return 0.0 if (np.isnan(v) or np.isinf(v)) else v
                    except (ValueError, TypeError):
                        return 0.0

                # --- STEP 1: AUTO-DETECT TIME STEP (dt) ---
                current_dt = 0.05  # Default 20Hz

                time_aliases = ["Time", "timestamp", "Date/Time", "Tid"]
                raw_time = self.get_column_data(df, time_aliases, optional=True)

                if raw_time is not None:
                    try:
                        # CRITICAL FIX: Handle European timestamps "10:00:01,500"
                        # We force convert to string, replace comma with dot, then parse.
                        time_str = raw_time.astype(str).str.replace(",", ".")
                        t_objs = pd.to_datetime(
                            time_str, errors="coerce", format="mixed"
                        )

                        # Calculate median difference between rows
                        deltas = t_objs.diff().dt.total_seconds().dropna()

                        # Filter out zeros (duplicate rows) and huge jumps
                        valid_deltas = deltas[(deltas > 0) & (deltas < 60)]

                        if len(valid_deltas) > 0:
                            # Take median of first 100 points to establish speed
                            calculated_dt = valid_deltas.head(100).median()

                            if calculated_dt > 0:
                                current_dt = calculated_dt
                                hz = 1 / current_dt
                                print(
                                    f"   -> Detected Speed: {hz:.2f} Hz (step={current_dt:.4f}s)"
                                )
                    except Exception as e:
                        print(f"   -> Time parsing failed ({e}). Defaulting to 20Hz.")

                # --- STEP 2: MAP ALL NUMERIC COLUMNS ---
                # Convert all columns to numeric, skip time columns
                time_cols = set()
                for alias in time_aliases:
                    if alias in df.columns:
                        time_cols.add(alias)

                # Store column names (excluding time)
                numeric_columns = []
                column_data = {}

                for col in df.columns:
                    if col in time_cols:
                        continue
                    # Skip ghost columns created by trailing delimiters (e.g. "Unnamed: 10")
                    if str(col).startswith("Unnamed:"):
                        continue
                    # Try to convert to numeric
                    try:
                        column_data[col] = df[col].map(to_float)
                        numeric_columns.append(col)
                    except Exception:
                        # Skip non-numeric columns
                        continue

                print(
                    f"   -> Found {len(numeric_columns)} numeric columns: {numeric_columns[:5]}..."
                )

                # Also map the required columns for backward compatibility
                flow_aliases = [
                    "Flow rate (Mean)",
                    "Flow rate (Arith. Mean)",
                    "Flow rate",
                ]
                p_in_aliases = [
                    "TS inlet pressure (Mean)",
                    "Pressure after pump (Arith. Mean)",
                    "Pressure after pump",
                    "TS inlet pressure",
                ]
                p_out_aliases = [
                    "TS outlet pressure (Mean)",
                    "Pressure before pump (Arith. Mean)",
                    "Pressure before pump",
                    "TS outlet pressure",
                ]

                flow_col = self._find_column(df, flow_aliases)
                p_in_col = self._find_column(df, p_in_aliases)
                p_out_col = self._find_column(df, p_out_aliases)

                # Yield Data as dictionary with ALL columns
                # Parse actual timestamps for each row
                actual_times = None
                if raw_time is not None:
                    try:
                        time_str = raw_time.astype(str).str.replace(",", ".")
                        t_objs = pd.to_datetime(
                            time_str, errors="coerce", format="mixed"
                        )
                        if t_objs.notna().sum() > 0:
                            t0 = t_objs.dropna().iloc[0]
                            actual_times = (t_objs - t0).dt.total_seconds().values
                    except Exception:
                        actual_times = None

                # Yield Data as dictionary with ALL columns
                for i in range(len(df)):
                    # Use actual timestamp or synthetic
                    if (
                        actual_times is not None
                        and i < len(actual_times)
                        and not np.isnan(actual_times[i])
                    ):
                        row_time = global_time_counter + actual_times[i]
                    else:
                        row_time = global_time_counter + i * current_dt

                    # Build raw data dict with all columns
                    raw_data = {}
                    for col in numeric_columns:
                        raw_data[col] = column_data[col].iloc[i]

                    # Calculate derived values for backward compatibility
                    flow = column_data[flow_col].iloc[i] if flow_col else 0.0
                    p_in = column_data[p_in_col].iloc[i] if p_in_col else 0.0
                    p_out = column_data[p_out_col].iloc[i] if p_out_col else 0.0

                    yield {
                        "time": row_time,
                        "flow": flow,
                        "p_in": p_in,
                        "p_out": p_out,
                        "raw": raw_data,
                        "columns": numeric_columns,
                    }

                # Update global counter for next file
                if actual_times is not None:
                    valid = actual_times[~np.isnan(actual_times)]
                    if len(valid) > 0:
                        global_time_counter += valid[-1] + current_dt
                    else:
                        global_time_counter += len(df) * current_dt
                else:
                    global_time_counter += len(df) * current_dt

            except KeyError as e:
                print(f"Skipping {file_path} - Column missing: {e}")
                continue
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

    def _find_column(self, df, aliases) -> Optional[str]:
        """Find first matching column name from aliases."""
        for name in aliases:
            if name in df.columns:
                return name
        return None

    def get_column_data(self, df, aliases, optional=False):
        """Return the first column whose name matches one of ``aliases``.

        Returns the pandas Series for the first alias present in ``df``. If none
        match: returns ``None`` when ``optional`` is True, otherwise raises
        ``KeyError`` (used for columns the caller treats as required).
        """
        for name in aliases:
            if name in df.columns:
                return df[name]
        if optional:
            return None
        raise KeyError(f"Could not find any of: {aliases}")
