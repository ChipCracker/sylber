"""Hydra entry point for Sylber 2.0 content-encoder training (stages 1-4).

Usage:
    python -m sylber2.train_content --config-name content_stage1
    python -m sylber2.train_content --config-name content_stage2 \\
        model_ckpt=outputs/stage1/last.ckpt
"""
import os
import faulthandler
import torch
import hydra
import lightning as pl

if os.environ.get("SYLBER2_WATCHDOG"):
    # periodically dump all thread stacks to stderr to diagnose hangs
    faulthandler.dump_traceback_later(int(os.environ["SYLBER2_WATCHDOG"]),
                                      repeat=True)
from omegaconf import OmegaConf
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from sylber2.data.datamodules import ContentDataModule
from sylber2.training.content_trainer import ContentTrainer


def load_student_weights(model, ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state)
    sd = {k[len("net."):]: v for k, v in sd.items() if k.startswith("net.")} or sd
    sd = {k: v for k, v in sd.items() if k.startswith("student.")}
    missing, unexpected = model.net.load_state_dict(sd, strict=False)
    print(f"Loaded previous-stage student: {len(sd)} tensors "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")


@hydra.main(config_path="configs", config_name="content_stage1", version_base=None)
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.get("seed", 7))
    datamodule = ContentDataModule(**cfg.data)
    model = ContentTrainer(**cfg.model)
    if cfg.get("model_ckpt"):
        load_student_weights(model, cfg.model_ckpt)

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(every_n_train_steps=cfg.get("checkpoint_every_steps", 25000),
                        save_last=True, save_top_k=-1),
    ]
    trainer = pl.Trainer(
        devices=cfg.get("devices", 1),
        accelerator=cfg.get("accelerator", "gpu"),
        strategy=cfg.get("strategy", "auto"),
        precision=cfg.get("precision", "bf16-mixed"),
        max_steps=cfg.max_steps,
        num_sanity_val_steps=0,
        val_check_interval=cfg.get("val_check_interval", 5000),
        check_val_every_n_epoch=None,
        limit_val_batches=cfg.get("limit_val_batches", 10),
        callbacks=callbacks,
        gradient_clip_val=cfg.get("gradient_clip_val", 0.5),
        default_root_dir=cfg.get("name", "sylber2_content"),
        log_every_n_steps=50,
    )
    trainer.fit(model, datamodule, ckpt_path=cfg.get("resume_ckpt"))


if __name__ == "__main__":
    main()
