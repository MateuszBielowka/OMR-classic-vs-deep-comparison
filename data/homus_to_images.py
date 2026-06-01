"""
Konwertuje pliki HOMUS (.txt ze współrzędnymi pociągnięć) na obrazy PNG.

Format pliku HOMUS:
  linia 1  : etykieta klasy (np. "Quarter-Note")
  pozostałe: pociągnięcia jako "x,y;x,y;..." (puste linie = separator pociągnięć)

Wynik: data/HOMUS_images/<klasa>/<plik>.png
       białe symbole na czarnym tle (zgodnie z konwencją classify_symbol)
"""

import cv2
import numpy as np
from pathlib import Path

# ── Konfiguracja ──────────────────────────────────────────────────────────────
HOMUS_DIR  = Path(__file__).parent / 'HOMUS'          # katalog z plikami .txt
OUT_DIR    = Path(__file__).parent / 'HOMUS_images'   # katalog wyjściowy
IMG_SIZE   = 96        # rozmiar boku obrazu (px) – symbole będą przeskalowane
PADDING    = 8         # margines (px) wokół symbolu
LINE_WIDTH = 2         # grubość pociągnięcia (px)
WHITE_ON_BLACK = True  # True = białe symbole na czarnym tle (jak classify_symbol)


def parse_strokes(txt_path: Path) -> tuple[str, list[list[tuple[int,int]]]]:
    """
    Czyta plik HOMUS.
    Zwraca (etykieta, lista_pociągnięć).
    Każde pociągnięcie to lista punktów (x, y).
    """
    lines = txt_path.read_text(encoding='utf-8', errors='ignore').strip().splitlines()
    if not lines:
        return '', []

    label = lines[0].strip()
    strokes = []
    current = []

    for line in lines[1:]:
        line = line.strip()
        if not line:
            if current:
                strokes.append(current)
                current = []
            continue
        points = []
        for token in line.split(';'):
            token = token.strip()
            if not token:
                continue
            parts = token.split(',')
            if len(parts) >= 2:
                try:
                    points.append((int(parts[0]), int(parts[1])))
                except ValueError:
                    continue
        current.extend(points)

    if current:
        strokes.append(current)

    return label, strokes


def render_strokes(strokes: list[list[tuple[int,int]]],
                   img_size: int = IMG_SIZE,
                   padding: int = PADDING,
                   line_width: int = LINE_WIDTH) -> np.ndarray | None:
    """
    Rysuje pociągnięcia na obrazie img_size×img_size.
    Normalizuje współrzędne tak, żeby symbol wypełniał obraz z marginesem.
    Zwraca obraz uint8 (białe symbole, czarne tło) lub None jeśli brak punktów.
    """
    all_pts = [p for s in strokes for p in s]
    if not all_pts:
        return None

    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    w = x_max - x_min or 1
    h = y_max - y_min or 1

    canvas_size = img_size - 2 * padding
    scale = canvas_size / max(w, h)

    img = np.zeros((img_size, img_size), dtype=np.uint8)

    for stroke in strokes:
        if len(stroke) < 2:
            continue
        pts = [
            (
                int((p[0] - x_min) * scale) + padding,
                int((p[1] - y_min) * scale) + padding,
            )
            for p in stroke
        ]
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i+1], 255, line_width, cv2.LINE_AA)

    return img


def convert_all(homus_dir: Path, out_dir: Path):
    txt_files = list(homus_dir.rglob('*.txt'))
    print(f'Znaleziono {len(txt_files)} plików .txt')

    skipped = 0
    converted = 0
    label_counts: dict[str, int] = {}

    for i, txt_path in enumerate(txt_files):
        label, strokes = parse_strokes(txt_path)
        if not label or not strokes:
            skipped += 1
            continue

        img = render_strokes(strokes)
        if img is None:
            skipped += 1
            continue

        # Zapisz do out_dir/<etykieta>/<oryginalna_nazwa>.png
        safe_label = label.replace('/', '_').replace(' ', '_')
        dest_dir = out_dir / safe_label
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / (txt_path.stem + '.png')

        if WHITE_ON_BLACK:
            cv2.imwrite(str(dest_path), img)
        else:
            cv2.imwrite(str(dest_path), cv2.bitwise_not(img))

        label_counts[safe_label] = label_counts.get(safe_label, 0) + 1
        converted += 1

        if (i + 1) % 1000 == 0:
            print(f'  {i+1}/{len(txt_files)} ...')

    print(f'\nGotowe: {converted} obrazów, {skipped} pominiętych')
    print(f'Klasy ({len(label_counts)}):')
    for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f'  {lbl:<35s} {cnt:5d}')


if __name__ == '__main__':
    convert_all(HOMUS_DIR, OUT_DIR)
