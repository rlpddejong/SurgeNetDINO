# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Adapted from EoMT (https://github.com/tue-mps/eomt).
# Modified by Ronald de Jong, 2025.
# ---------------------------------------------------------------

"""Per-frame semantic inference for the (image) EoMT model.

For every validation clip, the last ``num_predict_frames`` frames are read from
the clip's zip and segmented **independently** (EoMT has no temporal state), in
the same windowed semantic-inference fashion as ``inference.ipynb`` and the
validation step. The colourised predictions, overlays and (where available)
ground truth are written under ``<save_dir>/inference_eomt/<clip>/``.

Run standalone via ``inference_eomt.sh`` or automatically at the end of training
(see ``main.py``). All arguments default to the values in the training config,
so the minimal invocation is ``python3 inference_eomt.py --config <cfg>``.
"""

import logging
import os
from collections import deque
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn import functional as F

from inference_metrics import (
    METRIC_FRAMES,
    MetricsCollector,
    collect_from_disk,
    patient_of,
)
from inference_utils import (
    assemble_video,
    base_arg_parser,
    build_data_module,
    build_model,
    find_checkpoint,
    list_test_clips,
    load_config,
    load_gt_per_pixel,
    read_last_frames,
    resolve_data_path,
    resolve_num_predict_frames,
    resolve_num_video_clips,
    resolve_save_dir,
    save_frame_outputs,
)


@torch.no_grad()
def infer_semantic_frame(model, data, frame, device):
    """Windowed semantic inference of a single frame.

    Returns the per-pixel argmax prediction at the frame's original resolution.
    """
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )

    imgs = [frame.to(device)]
    img_sizes = [frame.shape[-2:]]
    crops, origins = model.window_imgs_semantic(imgs)

    with autocast_ctx:
        mask_logits_per_layer, class_logits_per_layer = model(crops)

    mask_logits = F.interpolate(mask_logits_per_layer[-1], data.img_size, mode="bilinear")
    crop_logits = model.to_per_pixel_logits_semantic(
        mask_logits, class_logits_per_layer[-1]
    )
    logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)

    return logits[0].argmax(0).cpu().numpy()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = base_arg_parser("EoMT per-frame semantic inference.").parse_args()

    config = load_config(args.config)
    data_path = resolve_data_path(config, args.data_path)
    save_dir = resolve_save_dir(config, args.save_dir)
    num_predict_frames = resolve_num_predict_frames(config, args.num_predict_frames)
    num_video_clips = resolve_num_video_clips(config, args.num_video_clips)
    num_metric_frames = max(num_predict_frames, METRIC_FRAMES)
    ckpt_path = find_checkpoint(save_dir, args.ckpt_path)

    # Predictions live next to the checkpoint (inside its run folder); the
    # metrics Excel goes in that same run folder, alongside inference_eomt/ and
    # inference_videos/.
    run_dir = os.path.dirname(os.path.abspath(ckpt_path))
    out_base = os.path.join(run_dir, "inference_eomt")
    video_base = os.path.join(run_dir, "inference_videos")
    excel_path = os.path.join(run_dir, "metrics_eomt.xlsx")

    if args.metrics_only:
        num_classes = config["data"].get("init_args", {}).get("num_classes")
        logging.info("metrics-only: reading saved predictions from %s", out_base)
        collector, n_clips = collect_from_disk(out_base, num_classes)
        collector.write_excel(excel_path)
        logging.info("metrics-only: processed %d clip folders", n_clips)
        return

    device = torch.device(args.device)

    logging.info(
        "EoMT inference | ckpt=%s | last %d frames/clip | full-video for first %d clips",
        ckpt_path,
        num_predict_frames,
        num_video_clips,
    )

    data = build_data_module(config, data_path)
    model = build_model(config, data, ckpt_path, device)

    clips, mask_dir = list_test_clips(data)
    if args.max_clips is not None:
        clips = clips[: args.max_clips]

    os.makedirs(out_base, exist_ok=True)

    collector = MetricsCollector(data.num_classes)

    for clip_idx, clip in enumerate(clips):
        make_video = clip_idx < num_video_clips

        # Video clips: predict every frame (for the video). Others: only the last
        # num_metric_frames (enough for the metrics) and save num_predict_frames.
        frames = read_last_frames(clip, 0 if make_video else num_metric_frames)
        save_count = len(frames) if make_video else min(num_predict_frames, len(frames))
        save_from = len(frames) - save_count

        clip_dir = os.path.join(out_base, clip.stem)
        mask_zip_path = mask_dir / clip.name

        records = deque(maxlen=METRIC_FRAMES)
        for i, (frame_name, frame) in enumerate(frames):
            pred = infer_semantic_frame(model, data, frame, device)
            gt = load_gt_per_pixel(mask_zip_path, frame_name, pred.shape)
            if i >= save_from:
                save_frame_outputs(clip_dir, frame_name, frame, pred, gt)
            records.append(
                (pred.astype(np.uint8), None if gt is None else gt.astype(np.uint8))
            )
        collector.add_clip(patient_of(clip.stem), records)

        if make_video:
            assemble_video(
                os.path.join(clip_dir, "overlay"),
                os.path.join(video_base, f"{clip.stem}.mp4"),
                fps=args.fps,
            )
            logging.info("[eomt] %s: predicted %d frames + video", clip.stem, len(frames))
        else:
            logging.info("[eomt] %s: saved %d frames", clip.stem, save_count)

    collector.write_excel(excel_path)

    logging.info("Done. Predictions in %s, videos in %s", out_base, video_base)


if __name__ == "__main__":
    main()
