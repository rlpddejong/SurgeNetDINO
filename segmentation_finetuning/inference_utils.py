# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Adapted from EoMT (https://github.com/tue-mps/eomt).
# Modified by Ronald de Jong, 2025.
# ---------------------------------------------------------------

"""Helpers for the EoMT inference script (``inference_eomt.py``).

These cover the generic bits of inference: loading a (training) config, building
the data module and the trained Lightning model from it, reading the last
frames of a clip from its zip, and colouring / saving the per-frame
predictions. The prediction loop itself lives in ``inference_eomt.py``.
"""

import argparse
import glob
import importlib
import inspect
import logging
import os
import shutil
import subprocess
import zipfile

import numpy as np
import torch
import yaml
from PIL import Image

from datasets.RAMIE_semantic import CLASS_MAPPING, COLOR_PALETTE

IGNORE_INDEX = 255


# --------------------------------------------------------------------------- #
# Config / model / data loading
# --------------------------------------------------------------------------- #
def load_config(config_path: str) -> dict:
    """Load a training config. Handles both the hand-written ``configs/*.yaml``
    (``model`` / ``data`` / ``trainer`` at the top level) and the resolved
    ``config.yaml`` Lightning writes into the run dir."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "model" not in config:
        # Possibly wrapped under a subcommand key (e.g. {"fit": {...}}).
        for value in config.values():
            if isinstance(value, dict) and "model" in value:
                return value
        raise ValueError(f"Could not find a 'model' section in {config_path}")

    return config


def _import(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def _filter_kwargs(cls, kwargs: dict, drop=()):
    """Keep only kwargs the class constructor accepts, dropping ``drop`` (the
    args we pass explicitly). The resolved ``config.yaml`` Lightning writes
    inlines linked args (num_classes, img_size, ...) and defaults that the
    hand-written configs leave out, so a plain ``**init_args`` splat would raise
    duplicate / unexpected-keyword errors. Classes with ``**kwargs`` keep
    everything."""
    signature = inspect.signature(cls.__init__)
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )
    return {
        k: v
        for k, v in kwargs.items()
        if k not in drop and (has_var_kw or k in signature.parameters)
    }


def build_data_module(config: dict, data_path: str):
    """Instantiate and ``setup()`` the data module described by ``config``."""
    data_cls = _import(config["data"]["class_path"])

    # Set path / batch_size / num_workers / check_empty_targets explicitly; drop
    # them (and any non-constructor extras) so the splat can't clash.
    init_args = _filter_kwargs(
        data_cls,
        config["data"].get("init_args", {}),
        drop=("path", "batch_size", "num_workers", "check_empty_targets"),
    )

    data = data_cls(
        path=data_path,
        batch_size=1,
        num_workers=0,
        check_empty_targets=False,
        **init_args,
    ).setup()

    return data


def build_model(config: dict, data, ckpt_path: str, device):
    """Rebuild the trained Lightning model exactly as configured and load the
    checkpoint weights into it. Mirrors the construction in ``inference.ipynb``
    (the network class comes from the config, so no model-specific kwargs are
    injected)."""
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_cls = _import(encoder_cfg["class_path"])
    # Pass the trained checkpoint as the encoder's ckpt_path so it skips
    # downloading / loading the pretrained backbone weights (as at train time
    # with --model.ckpt_path); the checkpoint below provides all weights.
    encoder = encoder_cls(
        img_size=data.img_size,
        ckpt_path=ckpt_path,
        **_filter_kwargs(
            encoder_cls, encoder_cfg.get("init_args", {}), drop=("img_size", "ckpt_path")
        ),
    )

    network_cfg = config["model"]["init_args"]["network"]
    network_cls = _import(network_cfg["class_path"])
    network = network_cls(
        num_classes=data.num_classes,
        encoder=encoder,
        **_filter_kwargs(
            network_cls,
            network_cfg.get("init_args", {}),
            drop=("encoder", "num_classes", "img_size"),
        ),
    )

    lit_cls = _import(config["model"]["class_path"])
    # ckpt_path / delta_weights control *pretrained init* at train time; we load
    # the trained checkpoint ourselves below, so drop them to avoid a redundant
    # (and possibly missing-path) load.
    model_kwargs = _filter_kwargs(
        lit_cls,
        config["model"].get("init_args", {}),
        drop=(
            "network",
            "num_classes",
            "img_size",
            "ckpt_path",
            "delta_weights",
            "load_ckpt_class_head",
        ),
    )

    model = (
        lit_cls(
            img_size=data.img_size,
            num_classes=data.num_classes,
            network=network,
            **model_kwargs,
        )
        .eval()
        .to(device)
    )

    state_dict = _load_state_dict(ckpt_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logging.info(
        "Loaded checkpoint %s (params=%d, missing=%d, unexpected=%d)",
        ckpt_path,
        len(state_dict),
        len(missing),
        len(unexpected),
    )

    return model


def _load_state_dict(ckpt_path: str) -> dict:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        # Full Lightning checkpoints carry non-tensor objects (optimizer state,
        # callbacks, ...) that weights_only=True refuses to unpickle.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    return {k.replace("._orig_mod", ""): v for k, v in ckpt.items()}


# --------------------------------------------------------------------------- #
# Argument resolution
# --------------------------------------------------------------------------- #
def resolve_data_path(config: dict, cli_value):
    if cli_value:
        return cli_value
    data_path = config["data"].get("init_args", {}).get("path")
    if data_path:
        return data_path
    raise ValueError("No data path given (pass --data_path or set data.init_args.path)")


def resolve_save_dir(config: dict, cli_value):
    if cli_value:
        return cli_value
    logger_cfg = config.get("trainer", {}).get("logger", {})
    if isinstance(logger_cfg, dict):
        save_dir = logger_cfg.get("init_args", {}).get("save_dir")
        if save_dir:
            return save_dir
    raise ValueError(
        "No save dir given (pass --save_dir or set trainer.logger.init_args.save_dir)"
    )


def resolve_num_predict_frames(config: dict, cli_value, default: int = 25) -> int:
    if cli_value is not None:
        return cli_value
    value = config["data"].get("init_args", {}).get("num_predict_frames")
    return value if value is not None else default


def resolve_num_video_clips(config: dict, cli_value, default: int = 5) -> int:
    if cli_value is not None:
        return cli_value
    value = config["data"].get("init_args", {}).get("num_video_clips")
    return value if value is not None else default


def find_checkpoint(save_dir: str, cli_value):
    """Resolve the checkpoint path: an explicit ``--ckpt_path`` if given,
    otherwise the most recent ``best-*.ckpt`` (falling back to ``last.ckpt``)
    found anywhere under ``save_dir``."""
    if cli_value:
        if not os.path.exists(cli_value):
            raise FileNotFoundError(f"Checkpoint not found: {cli_value}")
        return cli_value

    candidates = glob.glob(os.path.join(save_dir, "**", "best-*.ckpt"), recursive=True)
    if not candidates:
        candidates = glob.glob(
            os.path.join(save_dir, "**", "last.ckpt"), recursive=True
        )
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found under {save_dir}. Pass --ckpt_path explicitly."
        )

    return max(candidates, key=os.path.getmtime)


# --------------------------------------------------------------------------- #
# Clip / frame IO
# --------------------------------------------------------------------------- #
def list_test_clips(data):
    """List the test clip zips and the matching mask dir (inference runs on the
    test split)."""
    if getattr(data, "test_dataset", None) is None:
        raise FileNotFoundError(
            "No test split found (expected MTL_split/test.split<N>.bundle). "
            "Inference runs on the test set."
        )
    return data.test_dataset.clips, data.test_dataset.mask_dir


def read_last_frames(zip_path, num_frames: int):
    """Return ``[(frame_name, uint8 (C, H, W) tensor)]`` for the last
    ``num_frames`` frames of a clip, oldest-first (annotated frame last)."""
    with zipfile.ZipFile(zip_path) as img_zip:
        names = sorted(n for n in img_zip.namelist() if n.endswith(".png"))
        selected = names[-num_frames:] if num_frames > 0 else names
        frames = []
        for name in selected:
            with img_zip.open(name) as file:
                arr = np.array(Image.open(file).convert("RGB"))
            frames.append((name, torch.from_numpy(arr).permute(2, 0, 1).contiguous()))
    return frames


def load_gt_per_pixel(mask_zip_path, frame_name, hw, ignore_index: int = IGNORE_INDEX):
    """Load the ground-truth segmentation for a single frame as a per-pixel
    class-id map (``hw`` = target ``(H, W)``), or ``None`` if no mask exists for
    that frame. Raw label ids are remapped with ``CLASS_MAPPING`` exactly as the
    dataset does."""
    if not os.path.exists(mask_zip_path):
        return None

    with zipfile.ZipFile(mask_zip_path) as mask_zip:
        if frame_name not in mask_zip.namelist():
            return None
        with mask_zip.open(frame_name) as file:
            raw = np.array(Image.open(file))

    if raw.ndim == 3:
        raw = raw[..., 0]

    mapped = np.zeros_like(raw, dtype=np.int64)
    for src, dst in CLASS_MAPPING.items():
        mapped[raw == src] = dst

    gt = torch.from_numpy(mapped)
    if tuple(gt.shape) != tuple(hw):
        gt = torch.nn.functional.interpolate(
            gt[None, None].float(), size=tuple(hw), mode="nearest"
        )[0, 0].long()

    return gt.numpy()


# --------------------------------------------------------------------------- #
# Colouring / saving
# --------------------------------------------------------------------------- #
def colorize(label_map: np.ndarray, ignore_index: int = IGNORE_INDEX) -> np.ndarray:
    """Map a per-pixel class-id array to an RGB image using the dataset's fixed
    ``COLOR_PALETTE`` (consistent colours across frames)."""
    h, w = label_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in np.unique(label_map):
        if class_id == ignore_index:
            continue
        rgb[label_map == class_id] = COLOR_PALETTE.get(int(class_id), (0, 0, 0))
    return rgb


def _overlay(img_hwc: np.ndarray, colored: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return (img_hwc.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha).astype(np.uint8)


def save_frame_outputs(clip_dir, frame_name, img_chw, pred, gt=None):
    """Save the colourised prediction, an overlay on the input frame, and (when
    available) the colourised ground truth for one frame.

    ``frame_name`` is the frame's original name in the dataset (zip); its
    basename is reused as the output filename. ``pred`` / ``gt`` are per-pixel
    class-id arrays at the input frame's resolution; ``img_chw`` is the uint8
    ``(C, H, W)`` input frame.
    """
    img_hwc = img_chw.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    colored_pred = colorize(pred)

    pred_dir = os.path.join(clip_dir, "pred")
    overlay_dir = os.path.join(clip_dir, "overlay")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    name = os.path.splitext(os.path.basename(frame_name))[0] + ".png"
    Image.fromarray(colored_pred).save(os.path.join(pred_dir, name))
    Image.fromarray(_overlay(img_hwc, colored_pred)).save(os.path.join(overlay_dir, name))

    if gt is not None:
        gt_dir = os.path.join(clip_dir, "gt")
        os.makedirs(gt_dir, exist_ok=True)
        Image.fromarray(colorize(gt)).save(os.path.join(gt_dir, name))


def assemble_video(overlay_dir, out_path, fps: int = 25):
    """Assemble the ``frame_*.png`` overlay sequence in ``overlay_dir`` into a
    video at ``out_path`` (``fps`` frames/second).

    Frames are taken in sorted filename order (the dataset's zero-padded frame
    names sort chronologically). Tries, in order: OpenCV, the ffmpeg binary,
    imageio, torchvision's PyAV writer, and finally an animated GIF. A failing
    backend just falls through to the next."""
    frame_files = sorted(glob.glob(os.path.join(overlay_dir, "*.png")))
    if not frame_files:
        logging.warning("No overlay frames in %s; skipping video.", overlay_dir)
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 1) OpenCV (opencv-python bundles its own ffmpeg -> mp4, no system install).
    try:
        import cv2

        first = cv2.imread(frame_files[0])  # BGR, which is what VideoWriter wants
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError("cv2.VideoWriter could not open the output file")
        for f in frame_files:
            writer.write(cv2.imread(f))
        writer.release()
        return
    except Exception as e:
        logging.warning("cv2 video writing failed for %s: %s", out_path, e)

    # 2) ffmpeg binary on the PNG sequence (best mp4 quality, no extra memory).
    # Use a system ffmpeg if present, else the static binary bundled with
    # imageio-ffmpeg (pip install imageio-ffmpeg) so no system install is needed.
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        try:
            import imageio_ffmpeg

            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_exe = None

    if ffmpeg_exe is not None:
        # Use a concat list so arbitrary (original) filenames work, not just a
        # frame_%04d sequence.
        list_path = out_path + ".frames.txt"
        try:
            with open(list_path, "w") as lf:
                for f in frame_files:
                    lf.write(f"file '{os.path.abspath(f)}'\n")
                    lf.write(f"duration {1.0 / fps}\n")
                lf.write(f"file '{os.path.abspath(frame_files[-1])}'\n")
            cmd = [
                ffmpeg_exe, "-y",
                "-f", "concat", "-safe", "0",
                "-i", list_path,
                "-vsync", "vfr",
                "-r", str(fps),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                out_path,
            ]
            result = subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            if result.returncode == 0:
                return
            logging.warning("ffmpeg failed (%d) for %s; trying other backends.", result.returncode, out_path)
        finally:
            if os.path.exists(list_path):
                os.remove(list_path)

    # 3) imageio (also uses imageio-ffmpeg under the hood).
    try:
        import imageio.v2 as imageio

        frames = [imageio.imread(f) for f in frame_files]
        imageio.mimwrite(out_path, frames, fps=fps, macro_block_size=None)
        return
    except Exception as e:
        logging.warning("imageio video writing failed for %s: %s", out_path, e)

    # 4) torchvision (PyAV backend).
    try:
        from torchvision.io import write_video

        frames = np.stack(
            [np.array(Image.open(f).convert("RGB")) for f in frame_files]
        )
        write_video(out_path, torch.from_numpy(frames), fps=fps)
        return
    except Exception as e:
        logging.warning("torchvision video writing failed for %s: %s", out_path, e)

    # 5) Animated GIF fallback (PIL only) — only if no mp4 backend is available.
    logging.warning(
        "No mp4 backend available for %s. Install opencv-python or "
        "imageio-ffmpeg for mp4 output. Writing GIF instead.",
        out_path,
    )
    try:
        gif_path = os.path.splitext(out_path)[0] + ".gif"
        imgs = [Image.open(f).convert("RGB") for f in frame_files]
        imgs[0].save(
            gif_path,
            save_all=True,
            append_images=imgs[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        logging.warning("Wrote GIF fallback instead of video: %s", gif_path)
    except Exception as e:
        logging.warning("All video backends failed for %s: %s", out_path, e)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def base_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config", required=True, help="Path to the (training) config YAML."
    )
    parser.add_argument(
        "--ckpt_path",
        default=None,
        help="Trained checkpoint. If omitted, the latest best-*.ckpt under "
        "--save_dir is used.",
    )
    parser.add_argument(
        "--data_path",
        default=None,
        help="Dataset root. Defaults to data.init_args.path from the config.",
    )
    parser.add_argument(
        "--save_dir",
        default=None,
        help="Output base dir. Defaults to trainer.logger.init_args.save_dir.",
    )
    parser.add_argument(
        "--num_predict_frames",
        type=int,
        default=None,
        help="Number of last frames per clip to predict. Defaults to "
        "data.init_args.num_predict_frames (else 25).",
    )
    parser.add_argument(
        "--num_video_clips",
        type=int,
        default=None,
        help="For the first k test clips, predict ALL frames and write an "
        "overlay video. Defaults to data.init_args.num_video_clips (else 5).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Frame rate of the overlay videos (default: 25).",
    )
    parser.add_argument(
        "--max_clips",
        type=int,
        default=None,
        help="Optionally limit the number of test clips processed.",
    )
    parser.add_argument(
        "--metrics_only",
        action="store_true",
        help="Skip inference and (re)compute metrics from the already-saved "
        "pred/gt frames under the run's inference folder.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cuda, cuda:0 or cpu.",
    )
    return parser
