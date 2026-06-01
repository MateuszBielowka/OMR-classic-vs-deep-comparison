"""
Buduje zbior danych symboli muzycznych z DeepScores V2.
Wynik: data/note_dataset/symbols/images/<id>.png
       data/note_dataset/symbols/labels.csv
"""

import csv
import cv2
import ijson
from pathlib import Path
from collections import defaultdict

# ── Konfiguracja ──────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent / 'archive' / 'ds2_dense'
IMG_DIR     = BASE_DIR / 'images'
OUT_DIR     = Path(__file__).parent / 'note_dataset' / 'symbols'
MAX_SAMPLES = 10_000
PAD_X, PAD_Y = 6, 8

# ── Definicje klas ────────────────────────────────────────────────────────────
ACCIDENTALS = {
    'accidentalSharp':       'sharp',
    'accidentalFlat':        'flat',
    'accidentalNatural':     'natural',
    'accidentalDoubleSharp': 'double_sharp',
    'accidentalDoubleFlat':  'double_flat',
}

NOTEHEADS = {
    'noteheadBlackOnLine', 'noteheadBlackInSpace',
    'noteheadHalfOnLine',  'noteheadHalfInSpace',
    'noteheadWholeOnLine', 'noteheadWholeInSpace',
}

SIMPLE_SYMBOLS = {
    'clefG':          'clef_treble',
    'clefF':          'clef_bass',
    'clefCAlto':      'clef_alto',
    'clefCTenor':     'clef_tenor',
    'clef8':          'clef_8',
    'restWhole':      'rest_whole',
    'restHalf':       'rest_half',
    'restQuarter':    'rest_quarter',
    'rest8th':        'rest_8th',
    'rest16th':       'rest_16th',
    'rest32nd':       'rest_32nd',
    'augmentationDot':'dot',
    'timeSig0':       'time_0',
    'timeSig1':       'time_1',
    'timeSig2':       'time_2',
    'timeSig3':       'time_3',
    'timeSig4':       'time_4',
    'timeSig5':       'time_5',
    'timeSig6':       'time_6',
    'timeSig7':       'time_7',
    'timeSig8':       'time_8',
    'timeSig9':       'time_9',
    'timeSigCommon':  'time_common',
    'timeSigCutCommon':'time_cut',
}

ALL_TARGETS = set(ACCIDENTALS) | set(SIMPLE_SYMBOLS) | {'staff'} | NOTEHEADS


# ── Geometria ─────────────────────────────────────────────────────────────────
def find_staff_for_bbox(bbox, staffs):
    cy = (bbox[1] + bbox[3]) / 2
    for s in staffs:
        if s[1] <= cy <= s[3]:
            return s
    return min(staffs, key=lambda s: abs((s[1]+s[3])/2 - cy))


def find_notehead_for_accidental(acc_bbox, noteheads, max_gap_x=40, max_gap_y=20):
    acc_cy = (acc_bbox[1] + acc_bbox[3]) / 2
    candidates = []
    for nh_bbox, nh_comments in noteheads:
        nh_cy = (nh_bbox[1] + nh_bbox[3]) / 2
        dx = nh_bbox[0] - acc_bbox[2]
        dy = abs(nh_cy - acc_cy)
        if -5 <= dx <= max_gap_x and dy <= max_gap_y:
            candidates.append((dx + dy * 0.5, nh_comments))
    if not candidates:
        return ''
    candidates.sort(key=lambda c: c[0])
    fields = dict(t.split(':') for t in candidates[0][1].split(';') if ':' in t)
    return fields.get('rel_position', '')


def geo_rel_position(bbox, staff):
    """Oblicz rel_position geometrycznie gdy brak danych z anotacji."""
    staff_cy = (staff[1] + staff[3]) / 2
    staff_h  = (staff[3] - staff[1]) or 1
    space    = staff_h / 8
    acc_cy   = (bbox[1] + bbox[3]) / 2
    return str(round((staff_cy - acc_cy) / (space / 2)))


def crop_with_staff(img, sym_bbox, staffs):
    """Wytnij symbol z kontekstem calej pieciolinii."""
    x1, y1s, x2, y2s = sym_bbox
    if staffs:
        staff = find_staff_for_bbox(sym_bbox, staffs)
        y1 = min(staff[1], y1s)
        y2 = max(staff[3], y2s)
    else:
        y1, y2 = y1s, y2s
    x1 = max(0, x1 - PAD_X);  x2 = min(img.shape[1], x2 + PAD_X)
    y1 = max(0, y1 - PAD_Y);  y2 = min(img.shape[0], y2 + PAD_Y)
    return img[y1:y2, x1:x2]


# ── Ekstrakcja per obraz ──────────────────────────────────────────────────────
def extract_symbols(img_id, by_img, id2img, id2cat):
    anns     = by_img[img_id]
    img_info = id2img[int(img_id)]
    img = cv2.imread(str(IMG_DIR / img_info['filename']), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    staffs, accidentals, noteheads, simples = [], [], [], []

    for a in anns:
        name = id2cat[a['cat_id'][0]]['name']
        if name not in ALL_TARGETS:
            continue
        bbox = [int(v) for v in a['a_bbox']]
        comments = a.get('comments', '')

        if name == 'staff':
            staffs.append(bbox)
        elif name in ACCIDENTALS:
            accidentals.append((bbox, name, comments))
        elif name in NOTEHEADS:
            noteheads.append((bbox, comments))
        elif name in SIMPLE_SYMBOLS:
            simples.append((bbox, name, comments))

    results = []

    # ── Znaki chromatyczne (z rel_position) ───────────────────────────────────
    for bbox, name, _ in accidentals:
        rel_pos = find_notehead_for_accidental(bbox, noteheads)
        if rel_pos == '' and staffs:
            rel_pos = geo_rel_position(bbox, find_staff_for_bbox(bbox, staffs))

        crop = crop_with_staff(img, bbox, staffs)
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            continue

        results.append({
            'crop':        crop,
            'symbol':      ACCIDENTALS[name],
            'cat_name':    name,
            'rel_position': rel_pos,
            'duration':    '',
        })

    # ── Proste symbole (klucze, pauzy, metrum, kropki) ────────────────────────
    for bbox, name, comments in simples:
        crop = crop_with_staff(img, bbox, staffs)
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            continue

        fields = dict(t.split(':') for t in comments.split(';') if ':' in t)
        results.append({
            'crop':        crop,
            'symbol':      SIMPLE_SYMBOLS[name],
            'cat_name':    name,
            'rel_position': fields.get('rel_position', ''),
            'duration':    fields.get('duration', ''),
        })

    return results


# ── Wczytanie JSON strumieniowo ───────────────────────────────────────────────
def load_json(json_path):
    id2img, id2cat, by_img = {}, {}, defaultdict(list)

    print('  images...')
    with open(json_path, 'rb') as f:
        for img in ijson.items(f, 'images.item'):
            id2img[img['id']] = img

    print('  categories...')
    with open(json_path, 'rb') as f:
        for cat_id, cat in ijson.kvitems(f, 'categories'):
            id2cat[cat_id] = cat

    print('  annotations...')
    with open(json_path, 'rb') as f:
        for _, a in ijson.kvitems(f, 'annotations'):
            name = id2cat[a['cat_id'][0]]['name']
            if name in ALL_TARGETS:
                by_img[a['img_id']].append(a)

    print(f'  {len(id2img)} obrazow, {len(id2cat)} kategorii')
    return id2img, id2cat, by_img


# ── Zapis ─────────────────────────────────────────────────────────────────────
def save(results, out_dir):
    img_dir = out_dir / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / 'labels.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'symbol', 'cat_name',
                                               'rel_position', 'duration'])
        writer.writeheader()
        for i, r in enumerate(results):
            fname = f'{i:06d}.png'
            cv2.imwrite(str(img_dir / fname), r['crop'])
            writer.writerow({
                'id':           fname,
                'symbol':       r['symbol'],
                'cat_name':     r['cat_name'],
                'rel_position': r['rel_position'],
                'duration':     r['duration'],
            })

    print(f'Zapisano {len(results)} obrazow -> {img_dir}')
    print(f'Etykiety -> {csv_path}')


def print_stats(results):
    from collections import Counter
    counts = Counter(r['symbol'] for r in results)
    print('\nRozklad symboli:')
    for sym, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f'  {sym:<20s} {cnt:6d}  ({100*cnt/len(results):.1f}%)')


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Wczytuje JSON...')
    id2img, id2cat, by_img = load_json(BASE_DIR / 'deepscores_train.json')

    print(f'\nEkstrakcja (limit: {MAX_SAMPLES})...')
    results = []
    for img_id in by_img:
        if len(results) >= MAX_SAMPLES:
            break
        results.extend(extract_symbols(img_id, by_img, id2img, id2cat))

    results = results[:MAX_SAMPLES]
    print_stats(results)
    save(results, OUT_DIR)
