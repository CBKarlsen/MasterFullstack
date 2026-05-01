# app.py
import tkinter as tk
from tkinter import filedialog, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from .backend import CloggingDetector
from .dataloader import DataStreamer


class CloggingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Modal Clogging Predictor")
        self.root.geometry("1400x950")  # Made it slightly wider for the extra info

        self.spectrum_window = None
        self.frame_skipper = 0

        # --- LEFT PANEL: CONTROLS ---
        control_frame = tk.Frame(root, width=280, bg="#e1e1e1")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(
            control_frame,
            text="Configuration",
            font=("Helvetica", 16, "bold"),
            bg="#e1e1e1",
        ).pack(pady=20)

        # --- CHECKBOXES ---
        tk.Label(
            control_frame,
            text="Active Methods:",
            bg="#e1e1e1",
            font=("Arial", 11, "bold"),
        ).pack(anchor="w", padx=10)

        self.vars = {
            "static": tk.BooleanVar(value=True),
            "composite": tk.BooleanVar(value=False),
            "turbulence": tk.BooleanVar(value=False),
            "ml": tk.BooleanVar(value=False),  # <--- NEW: AI Checkbox
        }

        tk.Checkbutton(
            control_frame,
            text="Hydraulic Anomaly (Static)",
            variable=self.vars["static"],
            bg="#e1e1e1",
            font=("Arial", 10),
        ).pack(anchor="w", padx=20)
        tk.Checkbutton(
            control_frame,
            text="Spectral Composite (Hybrid)",
            variable=self.vars["composite"],
            bg="#e1e1e1",
            font=("Arial", 10),
        ).pack(anchor="w", padx=20)
        tk.Checkbutton(
            control_frame,
            text="Pure Turbulence (Control)",
            variable=self.vars["turbulence"],
            bg="#e1e1e1",
            font=("Arial", 10),
        ).pack(anchor="w", padx=20)

        # AI Button (Blue to make it stand out)
        tk.Checkbutton(
            control_frame,
            text="AI / Machine Learning",
            variable=self.vars["ml"],
            bg="#e1e1e1",
            fg="blue",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", padx=20)

        ttk.Separator(control_frame, orient="horizontal").pack(fill="x", pady=20)

        # --- SPEED SLIDER ---
        tk.Label(
            control_frame,
            text="Simulation Speed:",
            bg="#e1e1e1",
            font=("Arial", 11, "bold"),
        ).pack(anchor="w", padx=10)
        self.speed_var = tk.IntVar(value=20)
        self.speed_slider = tk.Scale(
            control_frame,
            from_=1,
            to=200,
            orient=tk.HORIZONTAL,
            variable=self.speed_var,
            bg="#e1e1e1",
            highlightthickness=0,
        )
        self.speed_slider.pack(fill=tk.X, padx=10)

        ttk.Separator(control_frame, orient="horizontal").pack(fill="x", pady=20)

        btn_spectrum = tk.Button(
            control_frame,
            text="🔍 View Live Spectrum (Log-Log)",
            command=self.open_spectrum_window,
            bg="#673AB7",
            fg="white",
            font=("Arial", 10, "bold"),
        )
        btn_spectrum.pack(fill=tk.X, padx=10, pady=5)

        # --- PREDICTION BOX (STATIC) ---
        tk.Label(
            control_frame,
            text="Linear Prediction (Static):",
            bg="#e1e1e1",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", padx=10)
        self.pred_lbl = tk.Label(
            control_frame,
            text="Waiting...",
            bg="black",
            fg="#00ff00",
            font=("Courier", 14),
            height=2,
            relief="sunken",
        )
        self.pred_lbl.pack(fill=tk.X, padx=10, pady=5)

        # ==========================================
        # NEW: COMPOSITE PREDICTION (TRAFFIC LIGHT)
        # ==========================================
        tk.Label(
            control_frame,
            text="Composite Prediction (ETA):",
            bg="#e1e1e1",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", padx=10, pady=(15, 0))
        self.lbl_traffic_light = tk.Label(
            control_frame,
            text="System Stable",
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
            height=2,
        )
        self.lbl_traffic_light.pack(fill=tk.X, padx=10, pady=5)

        # ==========================================
        # NEW: PHYSICS DASHBOARD (SPECTRAL SLOPE)
        # ==========================================
        tk.Label(
            control_frame,
            text="Spectral Slope (β):",
            bg="#e1e1e1",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", padx=10, pady=(15, 0))
        self.lbl_slope = tk.Label(
            control_frame,
            text="β = 0.00",
            bg="white",
            fg="black",
            font=("Courier", 14, "bold"),
            relief="sunken",
        )
        self.lbl_slope.pack(fill=tk.X, padx=10, pady=5)

        # Legend
        legend_frame = tk.Frame(control_frame, bg="#e1e1e1")
        legend_frame.pack(fill=tk.X, padx=10)
        tk.Label(
            legend_frame,
            text="Healthy: < -2.5",
            fg="green",
            bg="#e1e1e1",
            font=("Arial", 8),
        ).pack(side=tk.LEFT)
        tk.Label(
            legend_frame,
            text="Clogged: > -1.5",
            fg="red",
            bg="#e1e1e1",
            font=("Arial", 8),
        ).pack(side=tk.RIGHT)

        ttk.Separator(control_frame, orient="horizontal").pack(fill="x", pady=20)

        # --- LOAD BUTTONS ---
        tk.Label(
            control_frame, text="Data Source:", bg="#e1e1e1", font=("Arial", 11, "bold")
        ).pack(anchor="w", padx=10)
        self.load_file_btn = tk.Button(
            control_frame,
            text="📄 Load Single File (Excel/CSV)",
            command=self.load_file,
            bg="#5c5c5c",
            fg="white",
        )
        self.load_file_btn.pack(fill=tk.X, padx=10, pady=5)
        self.load_dir_btn = tk.Button(
            control_frame,
            text="📂 Load Folder (Batch)",
            command=self.load_folder,
            bg="#5c5c5c",
            fg="white",
        )
        self.load_dir_btn.pack(fill=tk.X, padx=10, pady=5)

        self.status_lbl = tk.Label(
            control_frame, text="No source selected", fg="#666", bg="#e1e1e1"
        )
        self.status_lbl.pack(pady=5)

        self.start_btn = tk.Button(
            control_frame,
            text="▶ START SIMULATION",
            command=self.start_simulation,
            bg="#007acc",
            fg="white",
            font=("Arial", 12, "bold"),
            state="disabled",
        )
        self.start_btn.pack(fill=tk.X, padx=10, pady=20)

        # --- RIGHT PANEL: DASHBOARD ---
        self.fig = plt.Figure(figsize=(8, 8), dpi=100)
        self.fig.patch.set_facecolor("#f0f0f0")
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(
            side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10
        )

        self.detector = None
        self.streamer = None
        self.running = False

    def load_file(self):
        # NEW: Supports .csv AND .xlsx
        filename = filedialog.askopenfilename(
            filetypes=[("Data Files", "*.csv *.xlsx")]
        )
        if filename:
            self.filepath = filename
            self.status_lbl.config(text=f"File: {filename.split('/')[-1]}", fg="green")
            self.start_btn.config(state="normal", bg="#009933")

    def load_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.filepath = folder
            self.status_lbl.config(text=f"Folder: {folder.split('/')[-1]}", fg="green")
            self.start_btn.config(state="normal", bg="#009933")

    def start_simulation(self):
        if not hasattr(self, "filepath"):
            return

        self.active_methods = [key for key, var in self.vars.items() if var.get()]
        if not self.active_methods:
            self.status_lbl.config(text="Select at least one method!", fg="red")
            return

        self.detector = CloggingDetector()
        try:
            # DataStreamer automatically detects if self.filepath is a file or folder
            self.streamer = DataStreamer(self.filepath).stream()
        except Exception as e:
            self.status_lbl.config(text="Error loading data!", fg="red")
            print(e)
            return

        # Setup Axes
        self.fig.clf()
        total_plots = 1 + len(self.active_methods)
        self.axes = self.fig.subplots(total_plots, 1, sharex=True)
        if total_plots == 1:
            self.axes = [self.axes]

        self.ax_flow = self.axes[0]
        self.ax_flow.set_title("Live Flow Rate", fontweight="bold", fontsize=10)
        self.ax_flow.set_ylabel("Flow")
        self.ax_flow.grid(True, alpha=0.3)
        self.ax_flow.set_xlim(0, 14400)  # 14400
        self.ax_flow.set_ylim(0, 450)

        self.method_axes = {}
        for i, method in enumerate(self.active_methods):
            ax = self.axes[i + 1]
            ax.set_title(f"Method: {method.upper()}", fontsize=9)
            ax.set_ylabel("Score")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 200000)  # 14400
            if method == "composite":
                ax.set_yscale("log")
            self.method_axes[method] = ax

        self.fig.tight_layout()
        self.canvas.draw()

        self.data = {
            "time": [],
            "flow": [],
            "static": [],
            "composite": [],
            "turbulence": [],
        }
        self.baseline_buffer = []
        self.is_calibrating = True
        self.running = True

        # Disable load buttons while running
        self.load_file_btn.config(state="disabled")
        self.load_dir_btn.config(state="disabled")
        self.start_btn.config(text="⏹ STOP", command=self.stop_simulation, bg="#cc0000")

        self.run_step()

    def stop_simulation(self):
        self.running = False
        self.load_file_btn.config(state="normal")
        self.load_dir_btn.config(state="normal")
        self.start_btn.config(
            text="▶ RESTART", command=self.start_simulation, bg="#007acc"
        )

    def run_step(self):
        if not self.running:
            return
        try:
            loops = self.speed_var.get()
            latest_results = None  # To store the last result of this batch

            # --- 1. DATA PROCESSING LOOP (Pure Math, No GUI) ---
            for _ in range(loops):
                t, flow, p_in, p_out = next(self.streamer)
                dP = p_in - p_out

                if self.is_calibrating:
                    self.baseline_buffer.append(dP)
                    if len(self.baseline_buffer) >= 400:
                        self.detector.calibrate(self.baseline_buffer)
                        self.is_calibrating = False
                        print("Calibrated.")
                    results = {
                        "static": 0,
                        "composite": 1e-6,
                        "turbulence": 0,
                        "ml": 0,
                        "spectral_slope": 0,
                    }
                else:
                    results = self.detector.process_sample(dP, t)

                # Store data
                if results:
                    self.data["time"].append(t)
                    self.data["flow"].append(flow)

                    for m in self.active_methods:
                        if m == "ml":
                            self.data[m].append(results.get("ml_probability", 0))
                        else:
                            self.data[m].append(results.get(m, 0))

                    # Save for GUI update after the loop
                    latest_results = results

            # --- 2. GUI UPDATES (Once per step, not per loop) ---
            if latest_results:
                # A. Update Traffic Light
                if "light_color" in latest_results:
                    c_col = latest_results["light_color"]
                    c_msg = latest_results["status_msg"]
                    hex_col = {
                        "green": "#4CAF50",
                        "yellow": "#FFC107",
                        "red": "#F44336",
                    }.get(c_col, "gray")
                    self.lbl_traffic_light.config(text=c_msg, bg=hex_col)

                # B. Update Physics Slope Label
                if "spectral_slope" in latest_results:
                    beta = latest_results["spectral_slope"]
                    self.lbl_slope.config(text=f"β = {beta:.2f}")
                    if beta < -2.0:
                        self.lbl_slope.config(fg="#4CAF50")
                    elif beta < -1.5:
                        self.lbl_slope.config(fg="#FFC107")
                    else:
                        self.lbl_slope.config(fg="red")

                # C. Update Spectrum Window (Throttled: Every 10th frame)
                self.frame_skipper += 1
                if self.frame_skipper % 10 == 0:
                    if self.spectrum_window and tk.Toplevel.winfo_exists(
                        self.spectrum_window
                    ):
                        freqs = latest_results.get("raw_freqs")
                        power = latest_results.get("raw_spectrum")

                        if freqs is not None and len(freqs) > 1:
                            try:
                                self.ax_spec.cla()
                                self.ax_spec.loglog(
                                    freqs,
                                    power,
                                    "b-",
                                    color="#2196F3",
                                    alpha=0.8,
                                    lw=1.5,
                                )

                                # Slope Line
                                slope = latest_results.get("spectral_slope", 0)
                                if slope != 0:
                                    mid_idx = len(freqs) // 2
                                    mid_x, mid_y = freqs[mid_idx], power[mid_idx]
                                    y_line = mid_y * (freqs / mid_x) ** slope
                                    self.ax_spec.loglog(
                                        freqs,
                                        y_line,
                                        "r--",
                                        lw=1.5,
                                        alpha=0.8,
                                        label=f"β={slope:.2f}",
                                    )
                                    self.ax_spec.legend(loc="upper right")

                                self.ax_spec.set_title(
                                    "Live PSD (Kolmogorov)", fontweight="bold"
                                )
                                self.ax_spec.set_xlabel("Frequency (Hz)")
                                self.ax_spec.set_ylabel("Energy Density")
                                self.ax_spec.grid(
                                    True, which="both", ls="--", alpha=0.5
                                )
                                self.ax_spec.set_xlim(left=freqs[1], right=freqs[-1])
                                self.spec_canvas.draw()
                            except Exception as e:
                                print(f"Plot Error: {e}")

            # --- 3. MAIN DASHBOARD PLOTS (Downsampled) ---
            if len(self.data["time"]) > 5000:
                sl = slice(None, None, 10)
            else:
                sl = slice(None)

            t_data = self.data["time"][sl]

            # Plot Flow
            self.ax_flow.cla()
            self.ax_flow.plot(t_data, self.data["flow"][sl], "b-", lw=1)
            self.ax_flow.set_title("Live Flow Rate", fontweight="bold", fontsize=10)
            self.ax_flow.set_ylabel("Flow")
            self.ax_flow.grid(True, alpha=0.3)
            self.ax_flow.set_xlim(0, 20000)
            self.ax_flow.set_ylim(0, 450)

            # Plot Methods
            for method in self.active_methods:
                ax = self.method_axes[method]
                ax.cla()
                col = {
                    "static": "green",
                    "composite": "purple",
                    "turbulence": "gray",
                    "ml": "blue",
                }.get(method, "black")

                if method == "composite":
                    plot_data = np.maximum(self.data[method][sl], 1e-6)
                else:
                    plot_data = self.data[method][sl]

                ax.plot(t_data, plot_data, color=col, lw=1.2)
                ax.set_title(f"Method: {method.upper()}", fontsize=9)
                ax.set_ylabel("Score")
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 20000)
                if method == "composite":
                    ax.set_yscale("log")
                if method == "ml":
                    ax.set_ylim(0, 1.1)

                if method == "static" and not self.is_calibrating:
                    ax.axhline(
                        self.detector.critical_threshold,
                        color="red",
                        ls="--",
                        alpha=0.5,
                    )
                    # Static Prediction logic could go here if you want it back on graph

                if method == "composite" and not self.is_calibrating:
                    ax.axhline(
                        self.detector.fft_threshold, color="red", ls="--", alpha=0.5
                    )

            self.canvas.draw()
            self.root.after(10, self.run_step)

        except StopIteration:
            self.stop_simulation()
            self.status_lbl.config(text="Done", fg="blue")

    def open_spectrum_window(self):
        """Opens a separate window to visualize the live Log-Log spectrum."""
        # If window already exists, bring it to front instead of creating a new one
        if self.spectrum_window is not None and tk.Toplevel.winfo_exists(
            self.spectrum_window
        ):
            self.spectrum_window.lift()
            return

        # 1. Create the new window
        self.spectrum_window = tk.Toplevel(self.root)
        self.spectrum_window.title("Live Power Spectral Density (Kolmogorov Plot)")
        self.spectrum_window.geometry("700x600")

        # 2. Setup Matplotlib figure inside it
        fig_spec = plt.Figure(figsize=(7, 6), dpi=100)
        self.ax_spec = fig_spec.add_subplot(111)

        # Styling based on scientific papers
        self.ax_spec.set_title("Power Spectral Density (PSD)", fontweight="bold")
        self.ax_spec.set_xlabel("Frequency ($Hz$) [Log Scale]", fontsize=10)
        self.ax_spec.set_ylabel("Energy Density ($P^2/Hz$) [Log Scale]", fontsize=10)
        self.ax_spec.grid(
            True, which="both", ls="--", alpha=0.5
        )  # Grid for log decades

        fig_spec.tight_layout(pad=3.0)

        # 3. Embed canvas
        self.spec_canvas = FigureCanvasTkAgg(fig_spec, master=self.spectrum_window)
        self.spec_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = CloggingApp(root)
    root.mainloop()
