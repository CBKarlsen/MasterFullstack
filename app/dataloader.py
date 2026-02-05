# dataloader.py
import pandas as pd
import os
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Generator


class ColumnMetadata:
    """Metadata about a column in the data file."""
    def __init__(self, name: str, col_type: str = "numeric", unit: str = ""):
        self.name = name
        self.col_type = col_type
        self.unit = unit

    def to_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "type": self.col_type,
            "unit": self.unit
        }


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
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath, sep=";", dtype=str, engine="python", nrows=10)
            elif filepath.endswith('.xlsx'):
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

            columns.append({
                "name": col,
                "type": "numeric",
                "unit": unit
            })

        return columns

    def stream(self) -> Generator[Dict[str, Any], None, None]:
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
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path, sep=";", dtype=str, engine="python")
                elif file_path.endswith('.xlsx'):
                    df = pd.read_excel(file_path, engine="calamine")
                else:
                    continue
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
                continue

            try:
                # Helper to clean numbers
                def to_float(x):
                    if isinstance(x, (int, float)): return float(x)
                    try:
                        return float(str(x).replace(',', '.'))
                    except:
                        return 0.0

                # --- STEP 1: AUTO-DETECT TIME STEP (dt) ---
                current_dt = 0.05 # Default 20Hz
                
                time_aliases = ["Time", "timestamp", "Date/Time", "Tid"]
                raw_time = self.get_column_data(df, time_aliases, optional=True)
                
                if raw_time is not None:
                    try:
                        # CRITICAL FIX: Handle European timestamps "10:00:01,500"
                        # We force convert to string, replace comma with dot, then parse.
                        time_str = raw_time.astype(str).str.replace(',', '.')
                        t_objs = pd.to_datetime(time_str, errors='coerce', format='mixed')
                        
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
                                print(f"   -> Detected Speed: {hz:.2f} Hz (step={current_dt:.4f}s)")
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
                    # Try to convert to numeric
                    try:
                        column_data[col] = df[col].map(to_float)
                        numeric_columns.append(col)
                    except Exception:
                        # Skip non-numeric columns
                        continue

                print(f"   -> Found {len(numeric_columns)} numeric columns: {numeric_columns[:5]}...")

                # Also map the required columns for backward compatibility
                flow_aliases = ["Flow rate (Mean)", "Flow rate (Arith. Mean)", "Flow rate"]
                p_in_aliases = ["TS inlet pressure (Mean)", "Pressure after pump (Arith. Mean)", "Pressure after pump", "TS inlet pressure"]
                p_out_aliases = ["TS outlet pressure (Mean)", "Pressure before pump (Arith. Mean)", "Pressure before pump", "TS outlet pressure"]

                flow_col = self._find_column(df, flow_aliases)
                p_in_col = self._find_column(df, p_in_aliases)
                p_out_col = self._find_column(df, p_out_aliases)

                # Yield Data as dictionary with ALL columns
                for i in range(len(df)):
                    # Build raw data dict with all columns
                    raw_data = {}
                    for col in numeric_columns:
                        raw_data[col] = column_data[col].iloc[i]

                    # Calculate derived values for backward compatibility
                    flow = column_data[flow_col].iloc[i] if flow_col else 0.0
                    p_in = column_data[p_in_col].iloc[i] if p_in_col else 0.0
                    p_out = column_data[p_out_col].iloc[i] if p_out_col else 0.0

                    yield {
                        "time": global_time_counter,
                        "flow": flow,
                        "p_in": p_in,
                        "p_out": p_out,
                        "raw": raw_data,
                        "columns": numeric_columns
                    }
                    global_time_counter += current_dt

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
        for name in aliases:
            if name in df.columns:
                return df[name]
        if optional: return None
        raise KeyError(f"Could not find any of: {aliases}")