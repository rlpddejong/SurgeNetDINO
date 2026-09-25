# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Adapted from EoMT (https://github.com/tue-mps/eomt).
# Modified by Ronald de Jong, 2025.
#
# Portions of this file are adapted from PyTorch Lightning,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------


import jsonargparse._typehints as _t
from types import MethodType
from gitignore_parser import parse_gitignore
import logging
import subprocess
import sys
import torch
import warnings
from datetime import datetime
from lightning.pytorch import cli
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.callbacks import (
    ModelSummary,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loops.training_epoch_loop import _TrainingEpochLoop
from lightning.pytorch.loops.fetchers import _DataFetcher, _DataLoaderIterDataFetcher

from training.lightning_module import LightningModule
from datasets.lightning_data_module import LightningDataModule

# Suppress PyTorch FX warnings for DINOv3 models
import os
os.environ["TORCH_LOGS"] = "-dynamo"


_orig_single = _t.raise_unexpected_value


def _raise_single(*args, exception=None, **kwargs):
    if isinstance(exception, Exception):
        raise exception
    return _orig_single(*args, exception=exception, **kwargs)


_orig_union = _t.raise_union_unexpected_value


def _raise_union(subtypes, val, vals):
    for e in reversed(vals):
        if isinstance(e, Exception):
            raise e
    return _orig_union(subtypes, val, vals)


_t.raise_unexpected_value = _raise_single
_t.raise_union_unexpected_value = _raise_union


# Name of this run's folder: the start time (e.g. "2026-06-10_11-36-50"). The
# folder lives under trainer.default_root_dir (the results dir passed on the CLI)
# instead of the random version folder Lightning/W&B would otherwise create.
RUN_NAME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def run_dir_for(trainer):
    """<default_root_dir>/<RUN_NAME>, e.g. $DATA_PATH/results/2026-06-10_11-36-50."""
    return os.path.join(trainer.default_root_dir, RUN_NAME)


class SaveConfigToRunDir(SaveConfigCallback):
    """Write the full resolved config as config.yaml into the run dir right at the
    start of the run, so the folder exists immediately and the checkpoints saved
    later can be reused for inference."""

    def setup(self, trainer, pl_module, stage):
        if not trainer.is_global_zero or self.already_saved:
            return
        run_dir = run_dir_for(trainer)
        os.makedirs(run_dir, exist_ok=True)
        self.parser.save(
            self.config,
            os.path.join(run_dir, "config.yaml"),
            skip_none=False,
            overwrite=True,
            multifile=self.multifile,
        )
        self.already_saved = True


def _should_check_val_fx(self: _TrainingEpochLoop, data_fetcher: _DataFetcher) -> bool:
    if not self._should_check_val_epoch():
        return False

    is_infinite_dataset = self.trainer.val_check_batch == float("inf")
    is_last_batch = self.batch_progress.is_last_batch
    if is_last_batch and (
        is_infinite_dataset or isinstance(data_fetcher, _DataLoaderIterDataFetcher)
    ):
        return True

    if self.trainer.should_stop and self.trainer.fit_loop._can_stop_early:
        return True

    is_val_check_batch = is_last_batch
    if isinstance(self.trainer.limit_train_batches, int) and is_infinite_dataset:
        is_val_check_batch = (
            self.batch_idx + 1
        ) % self.trainer.limit_train_batches == 0
    elif self.trainer.val_check_batch != float("inf"):
        if self.trainer.check_val_every_n_epoch is not None:
            is_val_check_batch = (
                self.batch_idx + 1
            ) % self.trainer.val_check_batch == 0
        else:
            # added below to check val based on global steps instead of batches in case of iteration based val check and gradient accumulation
            is_val_check_batch = (
                self.global_step
            ) % self.trainer.val_check_batch == 0 and not self._should_accumulate()

    return is_val_check_batch


class LightningCLI(cli.LightningCLI):
    def __init__(self, *args, **kwargs):
        logging.getLogger().setLevel(logging.INFO)
        torch.set_float32_matmul_precision("medium")
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.suppress_errors = True
        warnings.filterwarnings(
            "ignore",
            message=r".*It is recommended to use .* when logging on epoch level in distributed setting to accumulate the metric across devices.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"^The ``compute`` method of metric PanopticQuality was called before the ``update`` method.*",
        )
        warnings.filterwarnings(
            "ignore", message=r"^Grad strides do not match bucket view strides.*"
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*functools.partial will be a method descriptor in future Python versions*",
        )

        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser):
        parser.add_argument("--compile_disabled", action="store_true")

        parser.link_arguments(
            "data.init_args.num_classes", "model.init_args.num_classes"
        )
        parser.link_arguments(
            "data.init_args.num_classes",
            "model.init_args.network.init_args.num_classes",
        )

        parser.link_arguments(
            "data.init_args.stuff_classes", "model.init_args.stuff_classes"
        )

        parser.link_arguments("data.init_args.img_size", "model.init_args.img_size")
        parser.link_arguments(
            "data.init_args.img_size", "model.init_args.network.init_args.img_size"
        )
        parser.link_arguments(
            "data.init_args.img_size",
            "model.init_args.network.init_args.encoder.init_args.img_size",
        )

        parser.link_arguments(
            "model.init_args.ckpt_path",
            "model.init_args.network.init_args.encoder.init_args.ckpt_path",
        )

    def fit(self, model, **kwargs):
        if hasattr(self.trainer.logger.experiment, "log_code"):
            is_gitignored = parse_gitignore(".gitignore")
            include_fn = lambda path: path.endswith(".py") or path.endswith(".yaml")
            self.trainer.logger.experiment.log_code(
                ".", include_fn=include_fn, exclude_fn=is_gitignored
            )

        self.trainer.fit_loop.epoch_loop._should_check_val_fx = MethodType(
            _should_check_val_fx, self.trainer.fit_loop.epoch_loop
        )

        # Point checkpointing at <default_root_dir>/<RUN_NAME> (alongside the
        # config.yaml written by SaveConfigToRunDir). default_root_dir is only
        # known now, after the trainer has been built from the CLI args.
        for cb in self.trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                cb.dirpath = run_dir_for(self.trainer)

        if not self.config[self.config["subcommand"]]["compile_disabled"]:
            model = torch.compile(model)

        self.trainer.fit(model, **kwargs)

        self._run_inference_after_fit()

    def _run_inference_after_fit(self):
        """After training, run the EoMT inference script on the trained
        checkpoint, writing predictions under the logger's save_dir. This
        mirrors running inference_eomt.sh by hand; it runs as a separate process
        so it picks up the best checkpoint with a fresh model, independent of
        the (compiled) training model."""
        if not self.trainer.is_global_zero:
            return

        run_dir = run_dir_for(self.trainer)
        config_path = os.path.join(run_dir, "config.yaml")
        if not os.path.exists(config_path):
            logging.warning(
                "No config.yaml in %s; skipping post-training inference.", run_dir
            )
            return

        try:
            ckpt_path = ""
            for cb in self.trainer.callbacks:
                if isinstance(cb, ModelCheckpoint):
                    ckpt_path = cb.best_model_path or cb.last_model_path
                    break

            # Only used as the checkpoint-search root if no checkpoint is found
            # below; predictions themselves are written next to the checkpoint
            # (inside this run's folder) by the inference script.
            save_dir = getattr(self.trainer.logger, "save_dir", None) or run_dir

            # Release cached training memory before the inference subprocess.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            script_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "inference_eomt.py"
            )
            cmd = [
                sys.executable,
                script_path,
                "--config",
                config_path,
                "--save_dir",
                save_dir,
            ]
            if ckpt_path:
                cmd += ["--ckpt_path", ckpt_path]

            logging.info("Running post-training inference: %s", " ".join(cmd))
            subprocess.run(cmd, check=False)
        except Exception as e:  # inference must never fail the training job
            logging.warning("Post-training inference failed: %s", e)


def cli_main():
    LightningCLI(
        LightningModule,
        LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=SaveConfigToRunDir,
        seed_everything_default=0,
        trainer_defaults={
            "precision": "16-mixed",
            "enable_model_summary": False,
            "callbacks": [
                ModelSummary(max_depth=3),
                LearningRateMonitor(logging_interval="epoch"),
                ModelCheckpoint(
                    monitor="metrics/val_iou_all",
                    mode="max",
                    save_top_k=1,
                    filename="best-epoch{epoch:02d}",
                    auto_insert_metric_name=False,
                    save_last=True,
                ),
            ],
            "devices": 1,
            "gradient_clip_val": 0.01,
            "gradient_clip_algorithm": "norm",
        },
    )


if __name__ == "__main__":
    cli_main()
