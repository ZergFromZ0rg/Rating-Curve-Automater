import os
import tkinter as tk
from tkinter import filedialog, messagebox

import pandas as pd

from src.field_measurement_validation import clean_and_validate_measurements
from src.rating_curve_fitting import fit_rating_curve
from src.rating_curve_report import export_rating_curve_report


WINDOW_TITLE = "Rating Curve Automater"


class RatingCurveApp:
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry("900x760")
        self.root.configure(bg="#2d2d2d")
        self.root.resizable(False, False)

        self.dataset_path = tk.StringVar(value="")
        self.validation_summary = tk.StringVar(value="")
        self.fit_summary = tk.StringVar(value="")
        self.warning_messages = []
        self.cleaned_df = None
        self.fit_result = None

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

        dataset_label = tk.Label(page, text="Dataset:", bg="#2d2d2d", fg="white", font=("Helvetica", 14, "bold"))
        dataset_label.pack(anchor="w", padx=18)

        row = tk.Frame(page, bg="#2d2d2d")
        row.pack(fill="x", padx=18, pady=(8, 18))

        self.dataset_entry = tk.Entry(row, width=70, bg="#d9d9d9", fg="#1d1d1d", font=("Helvetica", 12))
        self.dataset_entry.pack(side="left", fill="x", expand=True)

        tk.Button(row, text="Browse", command=self._browse_dataset, width=12, bg="#a6a6a6", fg="#111111", font=("Helvetica", 12, "bold")).pack(side="left", padx=(12, 0))

        tk.Button(page, text="Upload and Validate Dataset", command=self._run_validation, width=34, height=2, bg="#b7b0b8", fg="#111111", font=("Helvetica", 13, "bold")).pack(fill="x", padx=18, pady=(8, 0))

        self.input_status = tk.Label(page, text="Ready", bg="#2d2d2d", fg="#f0f0f0", justify="left", font=("Helvetica", 11))
        self.input_status.pack(anchor="w", padx=18, pady=(24, 0))

        return page

    def _build_validation_page(self):
        page = tk.Frame(self.content, bg="#2d2d2d")

        tk.Label(page, text="Validation & Flags", bg="#2d2d2d", fg="white", font=("Helvetica", 24, "bold")).pack(pady=(18, 12))

        self.validation_label = tk.Label(page, textvariable=self.validation_summary, bg="#2d2d2d", fg="white", justify="left", font=("Helvetica", 12))
        self.validation_label.pack(anchor="w", padx=18, pady=(6, 10))

        tk.Label(page, text="Validation warnings:", bg="#2d2d2d", fg="white", font=("Helvetica", 12, "bold")).pack(anchor="w", padx=18)

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

        nav = tk.Frame(page, bg="#2d2d2d")
        nav.pack(fill="x", padx=18, pady=(6, 8))
        tk.Button(nav, text="Back", command=lambda: self.show_page("input"), width=16, bg="#b7b0b8", fg="#111111", font=("Helvetica", 11, "bold")).pack(side="left")
        tk.Button(nav, text="Continue with Log-Log Regression", command=self._run_fit, width=28, bg="#b7b0b8", fg="#111111", font=("Helvetica", 11, "bold")).pack(side="left", padx=(12, 0))

        return page

    def _build_export_page(self):
        page = tk.Frame(self.content, bg="#2d2d2d")

        tk.Label(page, text="Export Report", bg="#2d2d2d", fg="white", font=("Helvetica", 24, "bold")).pack(pady=(22, 18))

        self.fit_summary_label = tk.Label(page, textvariable=self.fit_summary, bg="#2d2d2d", fg="white", justify="left", font=("Helvetica", 12))
        self.fit_summary_label.pack(anchor="w", padx=18, pady=(8, 18))

        self.export_status = tk.Text(page, height=8, bg="#f3f3f3", fg="#151515", font=("Helvetica", 11), wrap="word")
        self.export_status.pack(fill="x", padx=18, pady=(0, 12))
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

    def _add_log_message(self, widget, message):
        widget.configure(state="normal")
        widget.insert("end", message + "\n")
        widget.see("end")
        widget.configure(state="disabled")

    def _set_warning_list(self, messages):
        self.warning_box.delete(0, "end")
        if not messages:
            self.warning_box.insert("end", "No warnings")
            return
        for item in messages:
            self.warning_box.insert("end", item)

    def _browse_dataset(self):
        path = filedialog.askopenfilename(title="Select dataset", filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")])
        if path:
            self.dataset_path.set(path)
            self.dataset_entry.delete(0, "end")
            self.dataset_entry.insert(0, path)
            self.input_status.config(text=f"Selected: {os.path.basename(path)}")

    def _run_validation(self):
        dataset = self.dataset_path.get().strip()
        if not dataset:
            self.input_status.config(text="No dataset selected. Please choose a file first.")
            return

        try:
            df = pd.read_excel(dataset, sheet_name="Measurements")
            cleaned = clean_and_validate_measurements(df)
            invalid = int((~cleaned["is_valid"]).sum())
            valid = int(cleaned["is_valid"].sum())

            self.cleaned_df = cleaned
            self.validation_summary.set(f"Valid rows: {valid}. Invalid rows: {invalid}.")

            flagged_rows = cleaned.loc[~cleaned["is_valid"], ["Date", "Stage Above Bed (m)", "Measured Discharge Q (m³/s)", "validation_notes"]]
            messages = []
            for _, row in flagged_rows.iterrows():
                messages.append(
                    f"Date: {row['Date']} | Stage: {row['Stage Above Bed (m)']} | Q: {row['Measured Discharge Q (m³/s)']} | {row['validation_notes']}"
                )
            self.warning_messages = messages
            self._set_warning_list(messages if messages else ["No warnings"])

            self.status_box.configure(state="normal")
            self.status_box.delete("1.0", "end")
            self.status_box.insert("end", f"Validation complete: {valid} valid rows, {invalid} invalid rows.\n")
            if messages:
                self.status_box.insert("end", "Validation warnings:\n")
                for item in messages:
                    self.status_box.insert("end", f"- {item}\n")
            else:
                self.status_box.insert("end", "No validation warnings detected.\n")
            self.status_box.configure(state="disabled")

            self.input_status.config(text="Dataset ready for review.")
            self.show_page("validation")

            if invalid > 0:
                response = messagebox.askyesno(
                    "Validation Warning",
                    "The dataset contains flagged or missing values. Do you want to continue with log-log regression?"
                )
                if not response:
                    self.status_box.configure(state="normal")
                    self.status_box.insert("end", "User stopped the process.\n")
                    self.status_box.configure(state="disabled")
                    return
                self.status_box.configure(state="normal")
                self.status_box.insert("end", "User chose to continue despite validation warnings.\n")
                self.status_box.configure(state="disabled")

        except Exception as exc:
            self.input_status.config(text=f"Validation failed: {exc}")
            self.warning_messages = [f"Validation failed: {exc}"]
            self._set_warning_list(self.warning_messages)
            self.status_box.configure(state="normal")
            self.status_box.delete("1.0", "end")
            self.status_box.insert("end", f"Validation failed: {exc}\n")
            self.status_box.configure(state="disabled")
            self.show_page("validation")

    def _run_fit(self):
        if self.cleaned_df is None:
            self.status_box.configure(state="normal")
            self.status_box.insert("end", "Please validate a dataset first.\n")
            self.status_box.configure(state="disabled")
            return

        try:
            self.status_box.configure(state="normal")
            self.status_box.delete("1.0", "end")
            self.status_box.insert("end", "Continuing with log-log regression.\n")
            self.status_box.configure(state="disabled")

            fit = fit_rating_curve(self.cleaned_df)
            self.fit_result = fit
            self.fit_summary.set(
                f"Fitted model: Q = {fit['a']:.6f} * (H - {fit['h0']:.3f})^{fit['b']:.6f} | R² = {fit['r_squared']:.4f}"
            )
            self.export_status.configure(state="normal")
            self.export_status.delete("1.0", "end")
            self.export_status.insert("end", f"Regression result: a={fit['a']:.6f}\n")
            self.export_status.insert("end", f"b={fit['b']:.6f}\n")
            self.export_status.insert("end", f"h0={fit['h0']:.3f}\n")
            self.export_status.insert("end", f"R²={fit['r_squared']:.4f}\n")
            self.export_status.configure(state="disabled")
            self.show_page("export")
        except Exception as exc:
            self.status_box.configure(state="normal")
            self.status_box.insert("end", f"Regression failed: {exc}\n")
            self.status_box.configure(state="disabled")

    def _export_report(self):
        if self.cleaned_df is None or self.fit_result is None:
            self.export_status.configure(state="normal")
            self.export_status.delete("1.0", "end")
            self.export_status.insert("end", "Please complete validation and regression before exporting.\n")
            self.export_status.configure(state="disabled")
            return

        output_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            title="Save Excel Report",
            filetypes=[("Excel files", "*.xlsx")],
        )
        if not output_path:
            return

        try:
            export_rating_curve_report(
                self.cleaned_df,
                output_path,
                a=self.fit_result["a"],
                b=self.fit_result["b"],
                h0=self.fit_result["h0"],
            )
            self.export_status.configure(state="normal")
            self.export_status.delete("1.0", "end")
            self.export_status.insert("end", f"Report exported successfully to:\n{output_path}\n")
            self.export_status.configure(state="disabled")
        except Exception as exc:
            self.export_status.configure(state="normal")
            self.export_status.delete("1.0", "end")
            self.export_status.insert("end", f"Export failed: {exc}\n")
            self.export_status.configure(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = RatingCurveApp(root)
    root.mainloop()
