"""Hydra entry point for Sylber 2.0 acoustic-encoder + vocoder training
(cycles 1-4, Appendix A.1.5).

Usage:
    python -m sylber2.train_vocoder --config-name vocoder_cycle1 \\
        model.content_ckpt=outputs/stage4/last.ckpt
    python -m sylber2.train_vocoder --config-name vocoder_cycle2 \\
        model.content_ckpt=... warm_start_ckpt=outputs/cycle1/last.ckpt
"""
import os
import faulthandler
import torch
import hydra
import lightning as pl

def _arm_watchdog():
    # periodically dump all thread stacks to stderr to diagnose hangs.
    # Called from main() only: at module level this would also run inside
    # every spawn dataloader worker (mp re-imports __main__) and flood the log.
    if os.environ.get("SYLBER2_WATCHDOG"):
        faulthandler.dump_traceback_later(int(os.environ["SYLBER2_WATCHDOG"]),
                                          repeat=True)
from omegaconf import OmegaConf
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from sylber2.data.datamodules import ResynthesisDataModule
from sylber2.training.vocoder_trainer import VocoderTrainer


@hydra.main(config_path="configs", config_name="vocoder_cycle1", version_base=None)
def main(cfg):
    _arm_watchdog()
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.get("seed", 7))
    datamodule = ResynthesisDataModule(**cfg.data)
    model = VocoderTrainer(**cfg.model)
    if cfg.get("warm_start_ckpt"):
        state = torch.load(cfg.warm_start_ckpt, map_location="cpu", weights_only=False)
        sd = state.get("state_dict", state)
        # keep everything except the (possibly absent) perceptual-loss weights
        sd = {k: v for k, v in sd.items() if not k.startswith("perceptual.")}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Warm start: missing={len(missing)} unexpected={len(unexpected)}")

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(every_n_train_steps=cfg.get("checkpoint_every_steps", 50000),
                        save_last=True, save_top_k=-1),
    ]
    trainer = pl.Trainer(
        devices=cfg.get("devices", 1),
        accelerator=cfg.get("accelerator", "gpu"),
        strategy=cfg.get("strategy", "auto"),
        precision=cfg.get("precision", "32-true"),
        max_steps=cfg.max_steps,
        num_sanity_val_steps=0,
        val_check_interval=cfg.get("val_check_interval", 10000),
        check_val_every_n_epoch=None,
        limit_val_batches=cfg.get("limit_val_batches", 10),
        callbacks=callbacks,
        default_root_dir=cfg.get("name", "sylber2_vocoder"),
        log_every_n_steps=50,
        enable_progress_bar=False,
    )
    trainer.fit(model, datamodule, ckpt_path=cfg.get("resume_ckpt"))


if __name__ == "__main__":
    main()
