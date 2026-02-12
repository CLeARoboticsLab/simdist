import datetime
import os
import yaml

import wandb
import torch
from torch.utils.data import random_split, get_worker_info, DataLoader
import flax.nnx as nnx
import orbax.checkpoint as ocp

from simdist.data.dataset import get_dataset, DatasetBase
from simdist.utils import io, model as model_utils, paths
from simdist.modeling import models


def train(cfg: dict):
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{date_time}" if cfg["run_name"] is None else cfg["run_name"]
    train_steps = 0

    max_steps = None
    if "max_steps" in cfg["training"] and cfg["training"]["max_steps"] is not None:
        if cfg["training"]["max_steps"] > 0:
            max_steps = cfg["training"]["max_steps"]
            print(f"Training will stop after {max_steps} steps.")

    # start wandb
    if cfg["wandb"]["log"]:
        wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"]["entity"],
            name=run_name,
            config=cfg,
        )
    print("Running training with config: ", cfg)

    # Create datasets
    dataset = get_dataset(cfg)
    generator = torch.Generator().manual_seed(cfg["training"]["seed"])
    train_dataset, test_dataset = random_split(
        dataset,
        [
            cfg["training"]["training_data_ratio"],
            1 - cfg["training"]["training_data_ratio"],
        ],
        generator=generator,
    )

    # holds the current training mode
    training = True

    # used to update each worker's dataset each time all of the data is seen
    def worker_init_fn(worker_id):
        worker_info = get_worker_info()
        if worker_info is not None:
            dataset: DatasetBase = worker_info.dataset.dataset
            if training:
                dataset.train()
            else:
                dataset.eval()

    # create dataloaders
    train_set = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        drop_last=True,
        shuffle=True,
        num_workers=cfg["data"]["num_train_workers"],
        worker_init_fn=worker_init_fn,
        # collate_fn=collate_to_numpy_safe,
        prefetch_factor=cfg["data"]["prefetch_factor"],
        persistent_workers=False,
        generator=generator,
    )
    test_set = DataLoader(
        test_dataset,
        batch_size=cfg["training"]["batch_size"],
        drop_last=True,
        num_workers=cfg["data"]["num_test_workers"],
        worker_init_fn=worker_init_fn,
        # collate_fn=collate_to_numpy_safe,
        prefetch_factor=cfg["data"]["prefetch_factor"],
        persistent_workers=False,
        generator=generator,
    )

    # load scaler params
    scaler_params = io.load_scaler_params(dataset.data_dir)

    # get the model
    ckpt_dir = None
    if (
        "resume_checkpoint" in cfg["checkpoint"]
        and cfg["checkpoint"]["resume_checkpoint"] is not None
    ):
        # load model from checkpoint if specified
        print("Resuming from checkpoint...")
        ckpt_dir = cfg["checkpoint"]["resume_checkpoint"]
        model, model_cfg, train_steps = model_utils.load_model_from_ckpt(ckpt_dir)
        cfg["model"] = model_cfg["model"]  # use the model cfg from the checkpoint
        if max_steps is not None:
            max_steps += train_steps
    else:
        # otherwise, create a new model
        print("Creating a new model...")
        model = models.get_model(
            cfg, scaler_params, rngs=nnx.Rngs(cfg["training"]["seed"])
        )

    # Setup checkpointing with Orbax
    if cfg["checkpoint"]["enabled"]:
        ckpt_options = ocp.CheckpointManagerOptions(
            max_to_keep=cfg["checkpoint"]["max_to_keep"],
            create=True,
            cleanup_tmp_directories=True,
            enable_async_checkpointing=False,
        )
        if ckpt_dir is None:
            ckpt_dir = os.path.join(paths.get_model_checkpoints_dir(), run_name)
            os.makedirs(ckpt_dir, exist_ok=True)
            with open(
                os.path.join(ckpt_dir, paths.get_model_config_filename()), "w"
            ) as f:
                struct = {}
                for k, v in scaler_params.items():
                    struct[k] = len(v["mean"])
                cfg["scaler_params_struct"] = struct
                yaml.dump(cfg, f, default_flow_style=False)

    # Get the loss function
    # TODO
