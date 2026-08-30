import os
import tkinter as tk
from tkinter import filedialog, messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.rating_curve_plot import make_rating_curve_figure
from src.workflow import (
    DEFAULT_SHEET_NAME,
    DEFAULT_UNCERTAINTY_THRESHOLD,
    RatingCurveWorkflow,
)


WINDOW_TITLE = "Rating Curve Automater"


class RatingCurveApp:
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry("900x820")
        self.root.configure(bg="#2d2d2d")
        self.root.resizable(False, False)

        self.workflow = RatingCurveWorkflow()

        self.validation_summary = tk.StringVar(value="")
        self.fit_summary = tk.StringVar(value="")
        self.log_scale_var = tk.BooleanVar(value=False)
        self.segments_var = tk.IntVar(value=1)

        self._build_ui()
        self.show_page("input")

    def _build_ui(self):
        self.top_bar = tk.Frame(self.root, bg="#393c3f", height=60)
        self.top_bar.pack(fill="x")
        tk.Label(self.top_bar, text="Rating Curve Automater", bg="#393c3f", fg="white", font=("Helvetica", 22, "bold")).pack(anchor="center", pady=12)

        self.content = tk.Frame(self.root, bg="#2d2d2d")
        self.content.pack(fill="both", expand=True, padx=25, pady=18)

        self.pages = {}
        self.pages["input"] = self._build_input_page()
        self.pages["validation"] = self._build_validation_page()
        self.pages["export"] = self._build_export_page()

    def _build_input_page(self):
        page = tk.Frame(self.content, bg="#2d2d2d")

        tk.Label(page, text="Input Dataset", bg="#2d2d2d", fg="white", font=("Helvetica", 26, "bold")).pack(pady=(30, 26))

        tk.Label(page, text="Dataset:", bg="#2d2d2d", fg="white", font=("Helvetica", 14, "bold")).pack(anchor="w", padx=18)

        row = tk.Frame(page, bg="#2d2d2d")
        row.pack(fill="x", padx=18, pady=(8, 12))

        self.dataset_entry = tk.Entry(row, width=70, bg="#d9d9d9", fg="#1d1d1d", font=("Helvetica", 12))
        self.dataset_entry.pack(side="left", fill="x", expand=True)

        tk.Button(row, text="Browse", command=self._browse_dataset, width=12, bg="#a6a6a6", fg="#111111", font=("Helvetica", 12, "bold")).pack(side="left", padx=(12, 0))

        sheet_row = tk.Frame(page, bg="#2d2d2d")
        sheet_row.pack(fill="x", padx=18, pady=(0, 18))
        tk.Label(sheet_row, text="Sheet name:", bg="#2d2d2d", fg="white", font=("Helvetica", 12, "bold")).pack(side="left")
        self.sheet_entry = tk.Entry(sheet_row, width=24, bg="#d9d9d9", fg="#1d1d1d", font=("Helvetica", 12))
        self.sheet_entry.insert(0, DEFAULT_SHEET_NAME)
        self.sheet_entry.pack(side="left", padx=(8, 0))

        tk.Button(page, text="Upload and Validate Dataset", command=self._run_validation, width=34, height=2, bg="#b7b0b8", fg="#111111", font=("Helvetica", 13, "bold")).pack(fill="x", padx=18, pady=(8, 0))

        self.input_status = tk.Label(page, text="Ready", bg="#2d2d2d", fg="#f0f0f0", justify="left", font=("Helvetica", 11))
        self.input_status.pack(anchor="w", padx=18, pady=(24, 0))

        return page

    def _build_validation_page(self):
        page = tk.Frame(self.content, bg="#2d2d2d")

        tk.Label(page, text="Validation & Flags", bg="#2d2d2d", fg="white", font=("Helvetica", 24, "bold")).pack(pady=(18, 12))

        tk.Label(page, textvariable=self.validation_summary, bg="#2d2d2d", fg="white", justify="left", font=("Helvetica", 12)).pack(anchor="w", padx=18, pady=(6, 10))

        tk.Label(page, text="Flagged rows (invalid = excluded, warning = kept):", bg="#2d2d2d", fg="white", font=("Helvetica", 12, "bold")).pack(anchor="w", padx=18)

        warning_frame = tk.Frame(page, bg="#2d2d2d")
        warning_frame.pack(fill="both", padx=18, pady=(8, 14), expand=True)

        scrollbar = tk.Scrollbar(warning_frame)
        scrollbar.pack(side="right", fill="y")
        self.warning_box = tk.Listbox(warning_frame, bg="#f3f3f3", fg="#111111", font=("Helvetica", 11), yscrollcommand=scrollbar.set, activestyle="none")
        self.warning_box.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.warning_box.yview)

        self.status_box = tk.Text(page, height=7, bg="#f3f3f3", fg="#151515", font=("Helvetica", 11), wrap="word")
        self.status_box.pack(fill="x", padx=18, pady=(0, 12))
        self.status_box.configure(state="disabled")

        h0_row = tk.Frame(page, bg="#2d2d2d")
        h0_row.pack(fill="x", padx=18, pady=(6, 2))
        tk.Label(h0_row, text="h0 (stage of zero flow, m):", bg="#2d2d2d", fg="white", font=("Helvetica", 11, "bold")).pack(side="left")
        self.h0_entry = tk.Entry(h0_row, width=12, bg="#d9d9d9", fg="#1d1d1d", font=("Helvetica", 11))
        self.h0_entry.pack(side="left", padx=(8, 8))
        tk.Label(h0_row, text="leave blank to estimate from the data", bg="#2d2d2d", fg="#c8c8c8", font=("Helvetica", 10, "italic")).pack(side="left")

        seg_row = tk.Frame(page, bg="#2d2d2d")
        seg_row.pack(fill="x", padx=18, pady=(2, 2))
        tk.Label(seg_row, text="Segments:", bg="#2d2d2d", fg="white", font=("Helvetica", 11, "bold")).pack(side="left")
        tk.Radiobutton(seg_row, text="1 (single power law)", variable=self.segments_var, value=1,
                       bg="#2d2d2d", fg="white", selectcolor="#2d2d2d", activebackground="#2d2d2d",
                       activeforeground="white", font=("Helvetica", 10)).pack(side="left", padx=(8, 0))
        tk.Radiobutton(seg_row, text="2 (piecewise, auto breakpoint)", variable=self.segments_var, value=2,
                       bg="#2d2d2d", fg="white", selectcolor="#2d2d2d", activebackground="#2d2d2d",
                       activeforeground="white", font=("Helvetica", 10)).pack(side="left", padx=(8, 0))

        nav = tk.Frame(page, bg="#2d2d2d")
        nav.pack(fill="x", padx=18, pady=(6, 8))
        tk.Button(nav, text="Back", command=lambda: self.show_page("input"), width=16, bg="#b7b0b8", fg="#111111", font=("Helvetica", 11, "bold")).pack(side="left")
        tk.Button(nav, text="Continue with Log-Log Regression", command=self._run_fit, width=28, bg="#b7b0b8", fg="#111111", font=("Helvetica", 11, "bold")).pack(side="left", padx=(12, 0))

        return page

    def _build_export_page(self):
        page = tk.Frame(self.content, bg="#2d2d2d")

        tk.Label(page, text="Export Report", bg="#2d2d2d", fg="white", font=("Helvetica", 24, "bold")).pack(pady=(18, 10))

        tk.Label(page, textvariable=self.fit_summary, bg="#2d2d2d", fg="white", justify="left", font=("Helvetica", 12)).pack(anchor="w", padx=18, pady=(4, 8))

        self.figure = Figure(figsize=(6.6, 3.4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=page)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=18, pady=(0, 8))

        opts_row = tk.Frame(page, bg="#2d2d2d")
        opts_row.pack(fill="x", padx=18, pady=(0, 6))
        tk.Checkbutton(
            opts_row, text="Log-log axes", variable=self.log_scale_var, command=self._draw_preview,
            bg="#2d2d2d", fg="white", selectcolor="#2d2d2d", activebackground="#2d2d2d", activeforeground="white",
            font=("Helvetica", 10, "bold"),
        ).pack(side="left")
        tk.Label(opts_row, text="Uncertainty threshold (relative error):", bg="#2d2d2d", fg="white", font=("Helvetica", 10, "bold")).pack(side="left", padx=(18, 4))
        self.threshold_entry = tk.Entry(opts_row, width=8, bg="#d9d9d9", fg="#1d1d1d", font=("Helvetica", 10))
        self.threshold_entry.insert(0, str(DEFAULT_UNCERTAINTY_THRESHOLD))
        self.threshold_entry.pack(side="left")

        self.export_status = tk.Text(page, height=5, bg="#f3f3f3", fg="#151515", font=("Helvetica", 11), wrap="word")
        self.export_status.pack(fill="x", padx=18, pady=(0, 10))
        self.export_status.configure(state="disabled")

        nav = tk.Frame(page, bg="#2d2d2d")
        nav.pack(fill="x", padx=18, pady=(6, 8))
        tk.Button(nav, text="Back", command=lambda: self.show_page("validation"), width=16, bg="#b7b0b8", fg="#111111", font=("Helvetica", 11, "bold")).pack(side="left")
        tk.Button(nav, text="Export Excel Report", command=self._export_report, width=22, bg="#b7b0b8", fg="#111111", font=("Helvetica", 11, "bold")).pack(side="left", padx=(12, 0))

        return page

    def show_page(self, page_name):
        for name, frame in self.pages.items():
            if name == page_name:
                frame.pack(fill="both", expand=True)
            else:
                frame.pack_forget()

    def _log(self, widget, message, clear=False):
        widget.configure(state="normal")
        if clear:
            widget.delete("1.0", "end")
        widget.insert("end", message + "\n")
        widget.see("end")
        widget.configure(state="disabled")

    def _set_warning_list(self, items):
        self.warning_box.delete(0, "end")
        if not items:
            self.warning_box.insert("end", "No flags")
            return
        for item in items:
            self.warning_box.insert("end", item)

    def _browse_dataset(self):
        path = filedialog.askopenfilename(title="Select dataset", filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")])
        if path:
            self.dataset_entry.delete(0, "end")
            self.dataset_entry.insert(0, path)
            self.input_status.config(text=f"Selected: {os.path.basename(path)}")

    def _run_validation(self):
        dataset = self.dataset_entry.get().strip()
        if not dataset:
            self.input_status.config(text="No dataset selected. Please choose a file first.")
            return

        sheet_name = self.sheet_entry.get().strip() or DEFAULT_SHEET_NAME

        try:
            result = self.workflow.load_and_validate(dataset, sheet_name=sheet_name)
        except Exception as exc:
            self.input_status.config(text=f"Validation failed: {exc}")
            self._set_warning_list([f"Validation failed: {exc}"])
            self._log(self.status_box, f"Validation failed: {exc}", clear=True)
            self.show_page("validation")
            return

        self.validation_summary.set(result.summary_line())

        listed = [f"INVALID | {line}" for line in result.flags]
        listed += [f"WARNING | {line}" for line in result.warnings]
        self._set_warning_list(listed)

        self._log(self.status_box, result.summary_line(), clear=True)
        if result.flags:
            self._log(self.status_box, f"{len(result.flags)} invalid row(s) will be excluded from the fit.")
        if result.warnings:
            self._log(self.status_box, f"{len(result.warnings)} row(s) kept with warnings.")
        if not listed:
            self._log(self.status_box, "No validation flags detected.")

        self.input_status.config(text="Dataset ready for review.")
        self.show_page("validation")

        if result.has_blocking_issues:
            proceed = messagebox.askyesno(
                "Validation Warning",
                "The dataset contains flagged or missing values. Continue with log-log regression?",
            )
            self._log(
                self.status_box,
                "User chose to continue despite validation warnings."
                if proceed
                else "User stopped the process.",
            )

    def _run_fit(self):
        if self.workflow.cleaned_df is None:
            self._log(self.status_box, "Please validate a dataset first.")
            return

        h0_text = self.h0_entry.get().strip()
        if h0_text:
            try:
                h0_override = float(h0_text)
            except ValueError:
                self._log(self.status_box, f"Invalid h0 value: {h0_text!r}. Enter a number or leave blank.")
                return
        else:
            h0_override = None

        try:
            self._log(self.status_box, "Continuing with log-log regression.", clear=True)
            outcome = self.workflow.run_fit(h0=h0_override, segments=self.segments_var.get())
        except Exception as exc:
            self._log(self.status_box, f"Regression failed: {exc}")
            return

        params = outcome.params
        self.fit_summary.set(f"Fitted model: {params['equation']}  |  R² = {params['r_squared']:.4f}")
        self._log(self.export_status, outcome.summary_line(), clear=True)
        self._draw_preview()
        self.show_page("export")

    def _draw_preview(self):
        params = self.workflow.fit_params
        if params is None or self.workflow.cleaned_df is None:
            return
        make_rating_curve_figure(
            self.workflow.cleaned_df,
            a=params["a"],
            b=params["b"],
            h0=params["h0"],
            figure=self.figure,
            log_scale=self.log_scale_var.get(),
            fit=params,
        )
        self.canvas.draw()

    def _export_report(self):
        if self.workflow.fit_params is None:
            self._log(self.export_status, "Please complete validation and regression before exporting.", clear=True)
            return

        threshold_text = self.threshold_entry.get().strip()
        try:
            threshold = float(threshold_text) if threshold_text else DEFAULT_UNCERTAINTY_THRESHOLD
        except ValueError:
            self._log(self.export_status, f"Invalid threshold: {threshold_text!r}. Enter a number.", clear=True)
            return

        output_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            title="Save Excel Report",
            filetypes=[("Excel files", "*.xlsx")],
        )
        if not output_path:
            return

        try:
            self.workflow.export_report(output_path, uncertainty_threshold=threshold)
            self._log(self.export_status, f"Report exported successfully to:\n{output_path}", clear=True)
        except Exception as exc:
            self._log(self.export_status, f"Export failed: {exc}", clear=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = RatingCurveApp(root)
    root.mainloop()
