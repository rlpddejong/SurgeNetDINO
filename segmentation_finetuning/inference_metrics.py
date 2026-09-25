# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Adapted from EoMT (https://github.com/tue-mps/eomt).
# Modified by Ronald de Jong, 2025.
# ---------------------------------------------------------------

"""Post-inference metrics for the RAMIE inference scripts.

Computes, per patient and per class, written to a single multi-sheet Excel:

* **Dice** — pixel-aggregated over the patient's clips, on the *last (annotated)
  frame* of each clip: all true/false positive/negative pixels are summed per
  class across the patient's clips, then turned into a single Dice per class.
* **HD95** — 95th-percentile (symmetric, surface) Hausdorff distance, computed
  per last frame then averaged over the patient's clips.
* **Video consistency (VC8 / VC16)** — for the last C frames of a clip, the
  "common area" is the set of pixels whose ground-truth class is the same across
  all C frames (per class: the intersection of ``gt == c``). VC is the
  proportion of that common area where the *prediction* is also constant
  (intersection of ``pred == c``). Computed per clip, then averaged per patient.

A "patient" is the part of the clip/frame name before the first ``_``.
"""

import glob
import logging
import os
from collections import defaultdict, deque

import numpy as np
from PIL import Image

from datasets.RAMIE_semantic import CLASS_NAMES, COLOR_PALETTE

# Inverse of COLOR_PALETTE: packed RGB int -> class id. The palette is a
# bijection over the classes, so colourised pred/gt PNGs can be decoded exactly.
_INV_PALETTE = {
    (r << 16) | (g << 8) | b: cid for cid, (r, g, b) in COLOR_PALETTE.items()
}

# Clip lengths (in frames) used for the video-consistency metric.
VC_WINDOWS = (8, 16)
# How many trailing frames each clip must provide to the metrics (max VC window;
# Dice/HD95 only need the last frame, which is always included).
METRIC_FRAMES = max(VC_WINDOWS)


def patient_of(name) -> str:
    """Patient id = the part of the clip/frame name before the first '_'."""
    return os.path.basename(str(name)).split("_")[0]


def _hd95_binary(pred: np.ndarray, gt: np.ndarray):
    """Symmetric 95th-percentile Hausdorff distance (in pixels) between two
    binary masks, or ``None`` if either mask (or its surface) is empty."""
    from scipy.ndimage import binary_erosion, distance_transform_edt

    if not pred.any() or not gt.any():
        return None

    pred_surface = pred ^ binary_erosion(pred)
    gt_surface = gt ^ binary_erosion(gt)
    if not pred_surface.any() or not gt_surface.any():
        return None

    dist_to_gt = distance_transform_edt(~gt_surface)
    dist_to_pred = distance_transform_edt(~pred_surface)

    distances = np.concatenate(
        [dist_to_gt[pred_surface], dist_to_pred[gt_surface]]
    )
    return float(np.percentile(distances, 95))


class MetricsCollector:
    """Accumulates per-patient metric statistics across clips and writes the
    final Excel. Feed each clip with :meth:`add_clip`."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.class_ids = list(range(num_classes))

        # patient -> (num_classes, 3) array of [tp, fp, fn] pixel counts.
        self.dice = defaultdict(
            lambda: np.zeros((num_classes, 3), dtype=np.int64)
        )
        # patient -> class -> list of per-frame HD95 values.
        self.hd95 = defaultdict(lambda: defaultdict(list))
        # window -> patient -> class -> list of per-clip VC ratios.
        self.vc = {C: defaultdict(lambda: defaultdict(list)) for C in VC_WINDOWS}

    # ----------------------------------------------------------------- #
    def add_clip(self, patient: str, records):
        """``records``: chronological iterable of ``(pred, gt)`` per-pixel
        class-id arrays for the clip's trailing frames (``gt`` may be ``None``);
        the last entry is the annotated frame used for Dice / HD95."""
        records = list(records)
        if not records:
            return

        pred_last, gt_last = records[-1]
        if gt_last is not None:
            self._add_dice(patient, pred_last, gt_last)
            self._add_hd95(patient, pred_last, gt_last)

        for window in VC_WINDOWS:
            if len(records) >= window:
                frames = records[-window:]
                if all(gt is not None for _, gt in frames):
                    self._add_vc(patient, window, frames)

    def _add_dice(self, patient, pred, gt):
        acc = self.dice[patient]
        for c in self.class_ids:
            p = pred == c
            g = gt == c
            tp = int(np.logical_and(p, g).sum())
            acc[c, 0] += tp
            acc[c, 1] += int(p.sum()) - tp  # fp
            acc[c, 2] += int(g.sum()) - tp  # fn

    def _add_hd95(self, patient, pred, gt):
        for c in self.class_ids:
            value = _hd95_binary(pred == c, gt == c)
            if value is not None:
                self.hd95[patient][c].append(value)

    def _add_vc(self, patient, window, frames):
        preds = [p for p, _ in frames]
        gts = [g for _, g in frames]
        for c in self.class_ids:
            common = gts[0] == c
            for g in gts[1:]:
                common = common & (g == c)
            denom = int(common.sum())
            if denom == 0:
                continue  # class not stably present -> undefined for this clip
            consistent = common.copy()
            for p in preds:
                consistent = consistent & (p == c)
            self.vc[window][patient][c].append(int(consistent.sum()) / denom)

    # ----------------------------------------------------------------- #
    def _all_patients(self):
        patients = set(self.dice) | set(self.hd95)
        for window in VC_WINDOWS:
            patients |= set(self.vc[window])
        return sorted(patients)

    def _tables(self):
        """{sheet_name: {patient: {class_id: value}}} for every metric."""
        tables = {}

        dice = {}
        for patient, acc in self.dice.items():
            row = {}
            for c in self.class_ids:
                tp, fp, fn = (int(x) for x in acc[c])
                denom = 2 * tp + fp + fn
                row[c] = (2 * tp / denom) if denom > 0 else np.nan
            dice[patient] = row
        tables["Dice"] = dice

        tables["HD95"] = {
            patient: {
                c: (float(np.mean(values[c])) if values[c] else np.nan)
                for c in self.class_ids
            }
            for patient, values in self.hd95.items()
        }

        for window in VC_WINDOWS:
            tables[f"VC_C{window}"] = {
                patient: {
                    c: (float(np.mean(values[c])) if values[c] else np.nan)
                    for c in self.class_ids
                }
                for patient, values in self.vc[window].items()
            }

        return tables

    def write_excel(self, out_path: str):
        tables = self._tables()
        patients = self._all_patients()
        class_names = [CLASS_NAMES.get(c, str(c)) for c in self.class_ids]

        if not patients:
            logging.warning("No metrics collected; skipping %s", out_path)
            return

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        try:
            from openpyxl import Workbook
        except Exception as e:
            logging.warning(
                "openpyxl not available (%s); writing CSVs instead of %s", e, out_path
            )
            self._write_csv(out_path, tables, patients, class_names)
            return

        wb = Workbook()
        wb.remove(wb.active)
        for sheet, table in tables.items():
            ws = wb.create_sheet(sheet[:31])
            for row in self._sheet_rows(table, patients, class_names):
                ws.append(row)
        wb.save(out_path)
        logging.info("Wrote metrics to %s", out_path)

    def _write_csv(self, out_path, tables, patients, class_names):
        import csv

        base = os.path.splitext(out_path)[0]
        for sheet, table in tables.items():
            path = f"{base}_{sheet}.csv"
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(self._sheet_rows(table, patients, class_names))
            logging.info("Wrote metrics CSV %s", path)

    def _sheet_rows(self, table, patients, class_names):
        """All rows of a sheet: header, one row per patient, then per-class
        ``mean`` and ``std`` summary rows across patients. The last column holds
        the per-patient average (excl. background); its summary cells are the
        mean / std of those per-patient averages."""
        rows = [["patient"] + class_names + ["average (excl. bg)"]]

        matrix, averages = [], []
        for patient in patients:
            values = [table.get(patient, {}).get(c, np.nan) for c in self.class_ids]
            average = self._class_average(values)
            matrix.append(values)
            averages.append(average)
            rows.append([patient] + [_cell(v) for v in values] + [_cell(average)])

        matrix = np.asarray(matrix, dtype=float).reshape(-1, len(self.class_ids))
        mean_per_class = [_nanmean(matrix[:, j]) for j in range(len(self.class_ids))]
        std_per_class = [_nanstd(matrix[:, j]) for j in range(len(self.class_ids))]

        rows.append(
            ["mean"] + [_cell(v) for v in mean_per_class] + [_cell(_nanmean(averages))]
        )
        rows.append(
            ["std"] + [_cell(v) for v in std_per_class] + [_cell(_nanstd(averages))]
        )
        return rows

    def _class_average(self, row):
        """Mean over classes, excluding the background class (id 0)."""
        values = [v for c, v in zip(self.class_ids, row) if c != 0]
        return _nanmean(values)


def _nanmean(values):
    arr = np.asarray(values, dtype=float)
    return np.nan if np.all(np.isnan(arr)) else float(np.nanmean(arr))


def _nanstd(values):
    arr = np.asarray(values, dtype=float)
    return np.nan if np.all(np.isnan(arr)) else float(np.nanstd(arr))


def _cell(value):
    """Format a metric value for a spreadsheet cell (NaN -> empty)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return round(float(value), 4)


def decolorize(rgb: np.ndarray) -> np.ndarray:
    """Invert :func:`inference_utils.colorize`: map an RGB image back to a
    per-pixel class-id array using the fixed palette (exact, lossless)."""
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    rgb = rgb[..., :3].astype(np.int64)
    key = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    out = np.zeros(key.shape, dtype=np.uint8)  # unknown colours -> 0 (background)
    for packed, cid in _INV_PALETTE.items():
        out[key == packed] = cid
    return out


def collect_from_disk(out_base: str, num_classes=None) -> "tuple[MetricsCollector, int]":
    """Build metrics from already-saved predictions instead of re-running
    inference. Reads the colourised ``pred/`` and ``gt/`` PNGs under each
    ``<out_base>/<clip>/`` folder, decoding them back to class ids.

    Returns the populated collector and the number of clip folders processed.
    """
    if num_classes is None:
        num_classes = len(CLASS_NAMES)

    collector = MetricsCollector(num_classes)
    if not os.path.isdir(out_base):
        logging.warning("Inference folder not found: %s", out_base)
        return collector, 0

    clip_dirs = sorted(
        d
        for d in glob.glob(os.path.join(out_base, "*"))
        if os.path.isdir(os.path.join(d, "pred"))
    )
    for clip_dir in clip_dirs:
        pred_files = sorted(glob.glob(os.path.join(clip_dir, "pred", "*.png")))
        if not pred_files:
            continue

        records = deque(maxlen=METRIC_FRAMES)
        for pred_file in pred_files[-METRIC_FRAMES:]:
            name = os.path.basename(pred_file)
            pred = decolorize(np.array(Image.open(pred_file).convert("RGB")))
            gt_file = os.path.join(clip_dir, "gt", name)
            gt = (
                decolorize(np.array(Image.open(gt_file).convert("RGB")))
                if os.path.exists(gt_file)
                else None
            )
            records.append((pred, gt))

        collector.add_clip(patient_of(os.path.basename(clip_dir)), records)

    return collector, len(clip_dirs)
