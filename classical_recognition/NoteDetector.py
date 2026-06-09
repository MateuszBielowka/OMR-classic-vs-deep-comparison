from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
GT_DIR = BASE_DIR / "images_gt2"
PRE_GT_DIR = BASE_DIR / "images_pre_gt"
OUTPUT_SYMBOL_TYPES_DIR = BASE_DIR / "output_symbol_types"
OUTPUT_NOTE_PITCHES_DIR = BASE_DIR / "output_note_pitches"
TARGET_HEIGHT = 150


@dataclass(frozen=True)
class SeparationParams:
    empty_slice_ratio: float = 0.04
    diff_slice_ratio: float = 0.01
    active_slice_min_ratio: float = 0.05
    active_slice_max_ratio: float = 0.20
    neighbor_diff_ratio: float = 0.20
    neighborhood_radius: int = 15
    min_symbol_width: int = 10


@dataclass
class ImageBundle:
    name: str
    source: np.ndarray
    gt: np.ndarray
    resized_source: np.ndarray
    resized_gt: np.ndarray
    processed: np.ndarray
    slice_sums: np.ndarray
    diff_sums: np.ndarray
    gt_columns: np.ndarray


def remove_border_objects(img):
    img = img.copy()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)

    h, w = img.shape
    result = np.zeros_like(img)

    for i in range(1, num_labels):
        x, y, width, height, _ = stats[i]
        touches_border = x == 0 or y == 0 or x + width == w or y + height == h
        if not touches_border:
            result[labels == i] = 255

    return result


def remove_staff_lines_soft(img):
    _, thresh = remove_staff_lines_soft_with_intermediate(img)
    return thresh


def remove_staff_lines_soft_with_intermediate(img):
    edges = img
    mask = np.zeros_like(img)
    rows, cols = mask.shape
    max_width = rows // 10
    min_line_len = rows * 13 // 25
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=90,
        minLineLength=min_line_len,
        maxLineGap=2,
    )

    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            dx = x2 - x1
            dy = y2 - y1
            if abs(np.degrees(np.arctan2(dy, dx))) < 40:
                cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=3)

    for col in range(cols):
        count = 0
        start = None

        for row in range(rows):
            if mask[row, col] == 255:
                if count == 0:
                    start = row
                count += 1
            else:
                if count > max_width:
                    mask[start:row, col] = 0
                count = 0

        if count > max_width:
            mask[start:rows, col] = 0

    painted = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    _, thresh = cv2.threshold(painted, 90, 255, cv2.THRESH_BINARY)
    return painted, thresh


def resize_to_target_height(img, target_height=TARGET_HEIGHT):
    height, width = img.shape[:2]
    if height == target_height:
        return img.copy()

    res_factor = target_height / height
    new_width = max(1, int(round(width * res_factor)))
    return cv2.resize(img, (new_width, target_height))


def extract_gt_columns(gt_img):
    blue = gt_img[:, :, 0].astype(np.int16)
    green = gt_img[:, :, 1].astype(np.int16)
    red = gt_img[:, :, 2].astype(np.int16)
    red_pixels = (red > 150) & (red > green + 40) & (red > blue + 40)
    return red_pixels.any(axis=0)


def build_feature_arrays(processed):
    height, width = processed.shape
    if width < 5:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    diff_image = cv2.absdiff(processed[:, : width - 3], processed[:, 3:width])
    slice_sums = np.empty(width - 5, dtype=np.int64)
    diff_sums = np.empty(width - 5, dtype=np.int64)

    for x in range(width - 5):
        slice_sums[x] = int(np.sum(processed[:, x : x + 2]))
        diff_sums[x] = int(np.sum(diff_image[:, x : x + 2]))

    return slice_sums, diff_sums


def load_image_bundles():
    image_files = {path.name for path in IMAGES_DIR.glob("*.png")}
    gt_files = {path.name for path in GT_DIR.glob("*.png")}
    common_files = sorted(image_files & gt_files)

    bundles = []
    for name in common_files:
        source = cv2.imread(str(IMAGES_DIR / name), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(GT_DIR / name), cv2.IMREAD_COLOR)

        if source is None or gt is None:
            print(f"Skipping {name}: failed to read one of the images")
            continue

        if source.shape[:2] != gt.shape[:2]:
            print(
                f"Skipping {name}: source and GT dimensions differ ({source.shape[:2]} vs {gt.shape[:2]})"
            )
            continue

        resized_source = resize_to_target_height(source)
        scale_factor = resized_source.shape[0] / source.shape[0]
        resized_gt = cv2.resize(
            gt,
            (max(1, int(round(gt.shape[1] * scale_factor))), resized_source.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        thresholded = cv2.threshold(resized_source, 127, 255, cv2.THRESH_BINARY_INV)[1]
        processed = remove_staff_lines_soft(thresholded)
        # processed = thresholded
        slice_sums, diff_sums = build_feature_arrays(processed)
        gt_columns = extract_gt_columns(resized_gt)

        bundles.append(
            ImageBundle(
                name=name,
                source=source,
                gt=gt,
                resized_source=resized_source,
                resized_gt=resized_gt,
                processed=processed,
                slice_sums=slice_sums,
                diff_sums=diff_sums,
                gt_columns=gt_columns,
            )
        )

    return bundles


def dilate_1d_mask(mask, radius):
    if radius <= 0:
        return mask

    padded = np.pad(mask, (radius, radius), mode="constant", constant_values=False)
    expanded = np.zeros_like(mask, dtype=bool)

    for offset in range(radius * 2 + 1):
        expanded |= padded[offset : offset + mask.size]

    return expanded


def apply_separation(processed, slice_sums, diff_sums, params):
    height, width = processed.shape
    final = processed.copy()
    zeroed_columns = np.zeros(width, dtype=bool)
    original_zero_columns = np.all(processed == 0, axis=0)
    indexes = []
    max_slice_sum = height * 2 * 255

    if width < 5:
        return final, indexes, zeroed_columns

    for x in range(width - 5):
        slice_diff = diff_sums[x]
        slice_sum = slice_sums[x]

        if slice_sum < max_slice_sum * params.empty_slice_ratio:
            final[:, x] = 0
            zeroed_columns[x] = True
            continue

        in_active_range = (
            max_slice_sum * params.active_slice_min_ratio
            <= slice_sum
            <= max_slice_sum * params.active_slice_max_ratio
        )

        if slice_diff < max_slice_sum * params.diff_slice_ratio and in_active_range:
            paint = True
            window_start = max(0, x - params.neighborhood_radius)
            window_end = min(width - 5, x + params.neighborhood_radius)

            for x1 in range(window_start, window_end + 1):
                if diff_sums[x1] > max_slice_sum * params.neighbor_diff_ratio:
                    paint = False
                    break

            if paint:
                final[:, x] = 0
                zeroed_columns[x] = True


    separator_indexes = np.flatnonzero(zeroed_columns)
    if separator_indexes.size >= 2:
        for i in range(separator_indexes.size - 1):
            left = int(separator_indexes[i])
            right = int(separator_indexes[i + 1])
            gap_width = right - left - 1
            if 0 < gap_width < 5:
                final[:, left + 1 : right] = 0
                zeroed_columns[left + 1 : right] = True


    x = 0
    while x < width:
        if not zeroed_columns[x]:
            x += 1
            continue

        start = x
        while x + 1 < width and zeroed_columns[x + 1]:
            x += 1
        end = x

        created_mask = ~original_zero_columns[start : end + 1]
        created_width = int(np.count_nonzero(created_mask))
        if 0 < created_width < 5:
            restore_cols = np.where(created_mask)[0] + start
            final[:, restore_cols] = processed[:, restore_cols]
            zeroed_columns[restore_cols] = False

        x += 1

    frame_start = None
    for x in range(width - 1):
        if frame_start is None:
            if np.all(final[:, x] == 0) and np.any(final[:, x + 1] != 0):
                frame_start = x + 1
        else:
            if np.any(final[:, x] != 0) and np.all(final[:, x + 1] == 0):
                frame_end = x
                if frame_end - frame_start > params.min_symbol_width:
                    indexes.append((frame_start - 1, frame_end + 1))
                frame_start = None

    return final, indexes, zeroed_columns


def separate_symbols(img, params=SeparationParams()):
    processed = remove_staff_lines_soft(img)
    # processed = img
    slice_sums, diff_sums = build_feature_arrays(processed)
    return apply_separation(processed, slice_sums, diff_sums, params)


def score_prediction(predicted_columns, gt_columns):
    aligned_gt = dilate_1d_mask(gt_columns, radius=1)

    tp = int(np.count_nonzero(predicted_columns & aligned_gt))
    fp = int(np.count_nonzero(predicted_columns & ~aligned_gt))
    fn = int(np.count_nonzero(~predicted_columns & aligned_gt))
    tn = int(np.count_nonzero(~predicted_columns & ~aligned_gt))

    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def evaluate_params(bundles, params):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    per_image_scores = []

    for bundle in bundles:
        _, _, predicted_columns = apply_separation(
            bundle.processed,
            bundle.slice_sums,
            bundle.diff_sums,
            params,
        )

        score = score_prediction(predicted_columns, bundle.gt_columns)
        per_image_scores.append((bundle.name, score))

        total_tp += score["tp"]
        total_fp += score["fp"]
        total_fn += score["fn"]
        total_tn += score["tn"]

    accuracy = (total_tp + total_tn) / (
        total_tp + total_fp + total_fn + total_tn
    )
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "params": params,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_image_scores": per_image_scores,
    }


def tune_parameters(bundles):
    grid = {
        "empty_slice_ratio": [0.03, 0.04, 0.05],
        "diff_slice_ratio": [0.008, 0.01, 0.012],
        "active_slice_min_ratio": [0.04, 0.05, 0.06],
        "active_slice_max_ratio": [0.18, 0.20, 0.22],
        "neighbor_diff_ratio": [0.15, 0.20, 0.25],
        "neighborhood_radius": [10, 15, 20],
    }

    candidate_params = []

    def build_candidates(index, current_values, names):
        if index == len(names):
            candidate_params.append(SeparationParams(**current_values.copy()))
            return

        name = names[index]
        for value in grid[name]:
            current_values[name] = value
            build_candidates(index + 1, current_values, names)

    build_candidates(0, {}, list(grid.keys()))

    scored_results = [evaluate_params(bundles, params) for params in candidate_params]
    scored_results.sort(key=lambda result: result["f1"], reverse=True)
    return scored_results[0], scored_results


def extract_file_number(file_name):
    stem = Path(file_name).stem
    match = re.search(r"\d+", stem)
    return match.group(0) if match else stem


def save_outputs(bundles, params):
    OUTPUT_SYMBOL_TYPES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_NOTE_PITCHES_DIR.mkdir(parents=True, exist_ok=True)

    for bundle in bundles:
        file_number = extract_file_number(bundle.name)
        final, indexes, _ = apply_separation(bundle.processed, bundle.slice_sums, bundle.diff_sums, params)
        for symbol_index, interval in enumerate(indexes, start=1):
            start, end = interval
            symbol_width = end - start
            if not (13 < symbol_width < 70):
                continue
            symbol_image = final[:, start:end]
            note_pitch_image = bundle.resized_source[:, start:end]
            output_name = f"{file_number}_{symbol_index}.png"
            cv2.imwrite(str(OUTPUT_SYMBOL_TYPES_DIR / output_name), symbol_image)
            cv2.imwrite(str(OUTPUT_NOTE_PITCHES_DIR / output_name), note_pitch_image)


def save_pre_gt_test_outputs(params, limit=20):
    if not PRE_GT_DIR.exists():
        print(f"Skipping test output save: directory not found: {PRE_GT_DIR}")
        return

    candidates = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        candidates.extend(PRE_GT_DIR.glob(pattern))

    image_paths = sorted(candidates)[:limit]
    if not image_paths:
        print(f"Skipping test output save: no image files found in {PRE_GT_DIR}")
        return

    OUTPUT_SYMBOL_TYPES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_NOTE_PITCHES_DIR.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for image_path in image_paths:
        source = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if source is None:
            print(f"Skipping {image_path.name}: failed to read image")
            continue

        resized_source = resize_to_target_height(source)
        thresholded = cv2.threshold(resized_source, 127, 255, cv2.THRESH_BINARY_INV)[1]
        final, indexes, _ = separate_symbols(thresholded, params)

        file_number = extract_file_number(image_path.name)
        for symbol_index, interval in enumerate(indexes, start=1):
            start, end = interval
            symbol_width = end - start
            if not (13 < symbol_width < 70):
                continue
            symbol_image = final[:, start:end]
            note_pitch_image = resized_source[:, start:end]
            output_name = f"{file_number}_{symbol_index}.png"
            cv2.imwrite(str(OUTPUT_SYMBOL_TYPES_DIR / output_name), symbol_image)
            cv2.imwrite(str(OUTPUT_NOTE_PITCHES_DIR / output_name), note_pitch_image)
            saved_count += 1

    print(f"Saved {saved_count} test symbols from {len(image_paths)} selected test images")


def show_pre_gt_samples(params, limit=10):
    if not PRE_GT_DIR.exists():
        print(f"Skipping preview: directory not found: {PRE_GT_DIR}")
        return

    candidates = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        candidates.extend(PRE_GT_DIR.glob(pattern))

    image_paths = sorted(candidates)[:limit]
    if not image_paths:
        print(f"Skipping preview: no image files found in {PRE_GT_DIR}")
        return

    print(f"Showing {len(image_paths)} sample images from {PRE_GT_DIR}")
    for idx, image_path in enumerate(image_paths, start=1):
        source = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if source is None:
            print(f"Skipping {image_path.name}: failed to read image")
            continue

        resized_source = resize_to_target_height(source)
        thresholded = cv2.threshold(resized_source, 127, 255, cv2.THRESH_BINARY_INV)[1]
        after_hough, processed = remove_staff_lines_soft_with_intermediate(thresholded)
        slice_sums, diff_sums = build_feature_arrays(processed)
        final, _, _ = apply_separation(processed, slice_sums, diff_sums, params)

        preview = cv2.vconcat([resized_source, thresholded, after_hough, final])
        cv2.imshow(
            f"Sample {idx}: {image_path.name} | original -> binary_inv -> post_hough -> processed",
            preview,
        )
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key == 27:
            print("Preview interrupted by user (ESC)")
            break


def main():
    best_params = SeparationParams(
        empty_slice_ratio=0.04,
        diff_slice_ratio=0.012,
        active_slice_min_ratio=0.04,
        active_slice_max_ratio=0.2,
        neighbor_diff_ratio=0.2,
        neighborhood_radius=10,
        min_symbol_width=10,
    )

    print("Using fixed parameters:")
    print(best_params)

    bundles = load_image_bundles()
    if bundles:
        print(f"Loaded {len(bundles)} matching image pairs from images/images_gt")
        save_outputs(bundles, best_params)
    else:
        print("No matching source/GT image pairs found in images and images_gt")

    metrics = evaluate_params(bundles, best_params)

    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")

    save_pre_gt_test_outputs(best_params, limit=20)
    show_pre_gt_samples(best_params, limit=20)


if __name__ == "__main__":
    main()