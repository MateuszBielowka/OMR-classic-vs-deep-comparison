from __future__ import annotations

import csv
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

try:
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover - runtime dependency check
    Image = None
    ImageTk = None


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
MAX_IMAGE_SIZE = (420, 420)


class LabelToolApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("OMR Pair Label Tool")

        project_root = Path(__file__).resolve().parents[1]
        self.note_dir = project_root / "data" / "note_pitches_selected"
        self.symbol_dir = project_root / "data" / "symbol_types_selected"
        self.output_csv = Path(__file__).resolve().parent / "labels.csv"

        self.pairs = self._find_pairs()
        self.labels = self._load_existing_labels()
        self.index = 0

        self.note_photo: ImageTk.PhotoImage | None = None
        self.symbol_photo: ImageTk.PhotoImage | None = None

        self._build_ui()
        self._go_to_next_unlabeled(0)

    def _build_ui(self) -> None:
        self.root.geometry("1100x700")

        top_frame = tk.Frame(self.root)
        top_frame.pack(fill="x", padx=10, pady=(10, 5))

        self.progress_var = tk.StringVar(value="")
        progress_label = tk.Label(top_frame, textvariable=self.progress_var, font=("Segoe UI", 11, "bold"))
        progress_label.pack(side="left")

        images_frame = tk.Frame(self.root)
        images_frame.pack(fill="both", expand=True, padx=10, pady=10)

        left_frame = tk.LabelFrame(images_frame, text="note_pitches_selected", padx=8, pady=8)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

        right_frame = tk.LabelFrame(images_frame, text="symbol_types_selected", padx=8, pady=8)
        right_frame.pack(side="left", fill="both", expand=True, padx=(5, 0))

        self.note_image_label = tk.Label(left_frame)
        self.note_image_label.pack(fill="both", expand=True)

        self.symbol_image_label = tk.Label(right_frame)
        self.symbol_image_label.pack(fill="both", expand=True)

        inputs_frame = tk.Frame(self.root)
        inputs_frame.pack(fill="x", padx=10, pady=(0, 10))

        tk.Label(inputs_frame, text="Label 1 (note pitch):").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=3)
        self.note_entry = tk.Entry(inputs_frame, width=12, font=("Segoe UI", 12))
        self.note_entry.grid(row=0, column=1, sticky="w", pady=3)

        tk.Label(inputs_frame, text="Label 2 (symbol type):").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=3)
        self.symbol_entry = tk.Entry(inputs_frame, width=12, font=("Segoe UI", 12))
        self.symbol_entry.grid(row=1, column=1, sticky="w", pady=3)

        buttons_frame = tk.Frame(inputs_frame)
        buttons_frame.grid(row=0, column=2, rowspan=2, padx=(20, 0), sticky="w")

        tk.Button(buttons_frame, text="Save + Next (Enter)", command=self.save_and_next, width=18).pack(side="left", padx=(0, 8))
        tk.Button(buttons_frame, text="Skip", command=self.skip, width=12).pack(side="left")

        hint = tk.Label(self.root, text="Esc: skip | Enter: save and next")
        hint.pack(anchor="w", padx=10, pady=(0, 10))

        self.root.bind("<Return>", lambda _: self.save_and_next())
        self.root.bind("<Escape>", lambda _: self.skip())

    def _find_pairs(self) -> list[str]:
        if not self.note_dir.exists() or not self.symbol_dir.exists():
            missing = []
            if not self.note_dir.exists():
                missing.append(str(self.note_dir))
            if not self.symbol_dir.exists():
                missing.append(str(self.symbol_dir))
            message = "Brak katalogow z obrazami:\n" + "\n".join(missing)
            raise FileNotFoundError(message)

        note_files = {p.name for p in self.note_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}
        symbol_files = {p.name for p in self.symbol_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}

        pairs = sorted(note_files.intersection(symbol_files))
        if not pairs:
            raise RuntimeError("Nie znaleziono wspolnych plikow obrazow w obu katalogach.")

        return pairs

    def _load_existing_labels(self) -> dict[str, tuple[str, str]]:
        if not self.output_csv.exists():
            return {}

        result: dict[str, tuple[str, str]] = {}
        with self.output_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = (row.get("filename") or "").strip()
                note_label = (row.get("label_1") or "").strip()
                symbol_label = (row.get("label_2") or "").strip()
                if filename:
                    result[filename] = (note_label, symbol_label)
        return result

    def _save_all_labels(self) -> None:
        with self.output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "label_1", "label_2"])
            writer.writeheader()
            for name in sorted(self.labels.keys()):
                label_1, label_2 = self.labels[name]
                writer.writerow({"filename": name, "label_1": label_1, "label_2": label_2})

    def _load_image_for_widget(self, path: Path) -> ImageTk.PhotoImage:
        image = Image.open(path)
        image.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(image)

    def _update_view(self) -> None:
        if self.index >= len(self.pairs):
            self.progress_var.set(f"Gotowe. Oznaczone: {len(self.labels)} / {len(self.pairs)}")
            self.note_image_label.config(image="")
            self.symbol_image_label.config(image="")
            self.note_entry.delete(0, tk.END)
            self.symbol_entry.delete(0, tk.END)
            messagebox.showinfo("Koniec", "Wszystkie pary zostaly przejrzane.")
            return

        filename = self.pairs[self.index]
        note_path = self.note_dir / filename
        symbol_path = self.symbol_dir / filename

        self.note_photo = self._load_image_for_widget(note_path)
        self.symbol_photo = self._load_image_for_widget(symbol_path)

        self.note_image_label.config(image=self.note_photo)
        self.symbol_image_label.config(image=self.symbol_photo)

        saved = self.labels.get(filename, ("", ""))
        self.note_entry.delete(0, tk.END)
        self.note_entry.insert(0, saved[0])
        self.symbol_entry.delete(0, tk.END)
        self.symbol_entry.insert(0, saved[1])

        self.progress_var.set(
            f"{self.index + 1}/{len(self.pairs)} | plik: {filename} | zapisane: {len(self.labels)}"
        )
        self.note_entry.focus_set()

    def _go_to_next_unlabeled(self, start_index: int) -> None:
        self.index = start_index
        while self.index < len(self.pairs) and self.pairs[self.index] in self.labels:
            self.index += 1
        self._update_view()

    def save_and_next(self) -> None:
        if self.index >= len(self.pairs):
            return

        label_1 = self.note_entry.get().strip()
        label_2 = self.symbol_entry.get().strip()

        if not label_1 or not label_2:
            messagebox.showwarning("Brak labeli", "Wpisz oba labele przed zapisaniem.")
            return

        filename = self.pairs[self.index]
        self.labels[filename] = (label_1, label_2)
        self._save_all_labels()

        self.index += 1
        self._update_view()

    def skip(self) -> None:
        if self.index >= len(self.pairs):
            return
        self.index += 1
        self._update_view()


def main() -> None:
    if Image is None or ImageTk is None:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Brak zaleznosci",
            "Brakuje biblioteki Pillow. Zainstaluj ja poleceniem: pip install pillow",
        )
        root.destroy()
        return

    root = tk.Tk()
    try:
        app = LabelToolApp(root)
    except (FileNotFoundError, RuntimeError) as exc:
        messagebox.showerror("Blad uruchomienia", str(exc))
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
