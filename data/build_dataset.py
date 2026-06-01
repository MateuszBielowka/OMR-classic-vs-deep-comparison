"""
Buduje zbiór danych izolowanych nut z DeepScores V2.

Wynik:
  data/note_dataset/images/<id>.png   – wycinek nuty z pięciolinią
  data/note_dataset/labels.csv        – id, note_type, duration, rel_position, img_src
"""

import json
import csv
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Konfiguracja ──────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent / 'archive' / 'ds2_dense'
IMG_DIR     = BASE_DIR / 'images'
OUT_DIR     = Path(__file__).parent / 'note_dataset'
MAX_SAMPLES = 10_000
PAD_X       = 6
PAD_Y       = 8

NOTEHEADS = {
    'noteheadBlackOnLine', 'noteheadBlackInSpace',
    'noteheadHalfOnLine',  'noteheadHalfInSpace',
    'noteheadWholeOnLine', 'noteheadWholeInSpace',
}
FLAGS = {'flag8thUp', 'flag8thDown', 'flag16thUp', 'flag16thDown'}


# ── Geometria ─────────────────────────────────────────────────────────────────
def x_overlap(b1, b2, tol=3):
    return b1[0] - tol <= b2[2] and b2[0] - tol <= b1[2]

def merge_bboxes(bboxes):
    return (min(b[0] for b in bboxes), min(b[1] for b in bboxes),
            max(b[2] for b in bboxes), max(b[3] for b in bboxes))

def is_stem_for_notehead(nh, st, max_gap=8):
    if not x_overlap(nh, st, tol=3):
        return False
    return st[1] <= nh[3] + max_gap and st[3] >= nh[1] - max_gap

def is_flag_for_stem(st, fl, tol=5):
    if not x_overlap(st, fl, tol=tol):
        return False
    span = st[3] - st[1] or 1
    return fl[1] <= st[1] + span * 0.4 or fl[3] >= st[3] - span * 0.4

def find_staff_for_note(nh_bbox, staffs):
    cy = (nh_bbox[1] + nh_bbox[3]) / 2
    for s in staffs:
        if s[1] <= cy <= s[3]:
            return s
    return min(staffs, key=lambda s: abs((s[1]+s[3])/2 - cy))


# ── Parsowanie pola comments ──────────────────────────────────────────────────
def parse_comments(comments):
    fields = {}
    for token in comments.split(';'):
        if ':' in token:
            k, v = token.split(':', 1)
            fields[k.strip()] = v.strip()
    return fields


# ── Główna logika ─────────────────────────────────────────────────────────────
def extract_notes(_, id2img, id2cat, by_img, max_samples):
    results = []   # lista (crop_img, meta_dict)

    for img_id, anns in by_img.items():
        if len(results) >= max_samples:
            break

        img_info = id2img[int(img_id)]
        img = cv2.imread(str(IMG_DIR / img_info['filename']), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        noteheads, stems, flags, staffs = [], [], [], []
        for a in anns:
            name = id2cat[a['cat_id'][0]]['name']
            bbox = [int(v) for v in a['a_bbox']]
            if name in NOTEHEADS:
                noteheads.append((bbox, name, a.get('comments', '')))
            elif name == 'stem':
                stems.append(bbox)
            elif name in FLAGS:
                flags.append(bbox)
            elif name == 'staff':
                staffs.append(bbox)

        for nh_bbox, nh_name, comments in noteheads:
            if len(results) >= max_samples:
                break

            # Dopasuj stem
            matching_stems = [s for s in stems if is_stem_for_notehead(nh_bbox, s)]
            if len(matching_stems) > 1:
                cx = (nh_bbox[0] + nh_bbox[2]) / 2
                cy = (nh_bbox[1] + nh_bbox[3]) / 2
                matching_stems = [min(matching_stems,
                    key=lambda s: abs((s[0]+s[2])/2 - cx) + abs((s[1]+s[3])/2 - cy))]

            # Dopasuj flagę
            matching_flags = [f for s in matching_stems
                              for f in flags if is_flag_for_stem(s, f)]

            # Granice poziome (nuta + stem + flaga)
            all_bboxes = [nh_bbox] + matching_stems + matching_flags
            x1, _, x2, _ = merge_bboxes(all_bboxes)

            # Granice pionowe (staff + nuta — dla nut poza pięciolinią)
            if staffs:
                staff = find_staff_for_note(nh_bbox, staffs)
                y1 = min(staff[1], nh_bbox[1], *(s[1] for s in matching_stems)) if matching_stems else min(staff[1], nh_bbox[1])
                y2 = max(staff[3], nh_bbox[3], *(s[3] for s in matching_stems)) if matching_stems else max(staff[3], nh_bbox[3])
            else:
                _, y1, _, y2 = merge_bboxes(all_bboxes)

            # Padding i clip do obrazu
            x1 = max(0, x1 - PAD_X);  x2 = min(img.shape[1], x2 + PAD_X)
            y1 = max(0, y1 - PAD_Y);  y2 = min(img.shape[0], y2 + PAD_Y)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 5:
                continue

            # Typ nuty
            is_open = 'Half' in nh_name or 'Whole' in nh_name
            if not matching_stems:
                note_type = 'note1'
            elif len(matching_flags) >= 2:
                note_type = 'note16'
            elif matching_flags:
                note_type = 'note8'
            elif is_open:
                note_type = 'note2'
            else:
                note_type = 'note4'

            fields = parse_comments(comments)
            results.append({
                'crop':         crop,
                'note_type':    note_type,
                'duration':     fields.get('duration', ''),
                'rel_position': fields.get('rel_position', ''),
                'img_src':      img_info['filename'],
            })

    return results


def save_dataset(results, out_dir):
    img_dir = out_dir / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / 'labels.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'note_type', 'duration', 'rel_position', 'img_src'])
        writer.writeheader()

        for i, r in enumerate(results):
            fname = f'{i:06d}.png'
            cv2.imwrite(str(img_dir / fname), r['crop'])
            writer.writerow({
                'id':           fname,
                'note_type':    r['note_type'],
                'duration':     r['duration'],
                'rel_position': r['rel_position'],
                'img_src':      r['img_src'],
            })

    print(f'Zapisano {len(results)} obrazow -> {img_dir}')
    print(f'Etykiety -> {csv_path}')


def print_stats(results):
    from collections import Counter
    types = Counter(r['note_type'] for r in results)
    print('\nRozkład typów nut:')
    for t, c in sorted(types.items()):
        print(f'  {t:<10s} {c:6d}  ({100*c/len(results):.1f}%)')


def load_json_streaming(json_path):
    """
    Ładuje deepscores_train.json strumieniowo przez ijson.
    Zwraca (id2img, id2cat, by_img) bez trzymania całego pliku w RAM.
    """
    import ijson

    id2img = {}
    id2cat = {}
    by_img = defaultdict(list)

    json_path = str(json_path)

    # ── images ────────────────────────────────────────────────────────────────
    print('  Parsowanie images...')
    with open(json_path, 'rb') as f:
        for img in ijson.items(f, 'images.item'):
            id2img[img['id']] = img

    # ── categories ────────────────────────────────────────────────────────────
    print('  Parsowanie categories...')
    with open(json_path, 'rb') as f:
        # categories to dict { "1": {...}, "2": {...} }
        for cat_id, cat in ijson.kvitems(f, 'categories'):
            id2cat[cat_id] = cat

    # ── annotations ───────────────────────────────────────────────────────────
    print('  Parsowanie annotations...')
    with open(json_path, 'rb') as f:
        for ann_id, a in ijson.kvitems(f, 'annotations'):
            by_img[a['img_id']].append(a)

    print(f'  Załadowano: {len(id2img)} obrazów, {len(id2cat)} kategorii, '
          f'{sum(len(v) for v in by_img.values())} anotacji')
    return id2img, id2cat, by_img


if __name__ == '__main__':
    print('Wczytuję JSON strumieniowo (ijson)...')
    id2img, id2cat, by_img = load_json_streaming(BASE_DIR / 'deepscores_train.json')

    print(f'\nPrzetwarzam (limit: {MAX_SAMPLES})...')
    results = extract_notes(None, id2img, id2cat, by_img, MAX_SAMPLES)

    print_stats(results)
    save_dataset(results, OUT_DIR)
