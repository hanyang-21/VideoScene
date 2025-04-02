import os
from pathlib import Path
import warnings

import hydra
import torch
import wandb
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers.wandb import WandbLogger

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from mvsplat.src.config import load_typed_root_config
    from mvsplat.src.dataset.data_module import DataModule
    from mvsplat.src.global_cfg import set_cfg
    from mvsplat.src.loss import get_losses
    from mvsplat.src.misc.LocalLogger import LocalLogger
    from mvsplat.src.misc.step_tracker import StepTracker
    from mvsplat.src.misc.wandb_tools import update_checkpoint_path
    from mvsplat.src.model.decoder import get_decoder
    from mvsplat.src.model.encoder import get_encoder
    from mvsplat.src.model.model_wrapper import ModelWrapper


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    # Set up the output directory.
    if cfg_dict.output_dir is None:
        output_dir = Path(
            hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
        )
    else:  # for resuming
        output_dir = Path(cfg_dict.output_dir)
        os.makedirs(output_dir, exist_ok=True)

    print(cyan(f"Saving outputs to {output_dir}."))
    
    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=None,
        devices="auto",
        num_nodes=cfg.trainer.num_nodes,
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        callbacks=None,
        val_check_interval=cfg.trainer.val_check_interval,
        enable_progress_bar=cfg.mode == "test",
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)

    model_kwargs = {
        "optimizer_cfg": cfg.optimizer,
        "test_cfg": cfg.test,
        "train_cfg": cfg.train,
        "encoder": encoder,
        "encoder_visualizer": encoder_visualizer,
        "decoder": get_decoder(cfg.model.decoder, cfg.dataset),
        "losses": get_losses(cfg.loss),
        "step_tracker": step_tracker,
        "output_path": output_dir
    }
    if cfg.mode == "train" and checkpoint_path is not None and not cfg.checkpointing.resume:
        # Just load model weights, without optimizer states
        # e.g., fine-tune from the released weights on other datasets
        model_wrapper = ModelWrapper.load_from_checkpoint(
            checkpoint_path, **model_kwargs, strict=True)
        print(cyan(f"Loaded weigths from {checkpoint_path}."))
    else:
        model_wrapper = ModelWrapper(**model_kwargs)

    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )

    if cfg.mode == "train":
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=(
            checkpoint_path if cfg.checkpointing.resume else None))
    else:
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')

    train()