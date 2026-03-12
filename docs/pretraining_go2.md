## Pretraining (Go2)

- [Expert Policy Training](#expert-policy-training)
- [Data Generation](#data-generation)
- [World Model Pretraining](#world-model-pretraining)
- [Deployment (Simulation)](#deployment-simulation)

### Expert Policy Training

To train, use the below, replacing `<wandb_username>` with your Weights and Biases username (if you don't have an account, create one at [https://wandb.ai/](https://wandb.ai/)). Runs are saved to `checkpoints/rl/`. The environment config is found in [`simdist/rl/go2.py`](simdist/rl/go2.py) and the PPO config is found in [`simdist/rl/rsl_rl_ppo_cfg.py`](simdist/rl/rsl_rl_ppo_cfg.py).

```bash
WANDB_USERNAME=<wandb_username> python scripts/train_rl.py --task Go2 --headless
```

To visualize and play an expert policy, use the below, replacing `<run_folder_name>` with the name of the folder from training that is found in `checkpoints/rl/` (for example: `2025-03-14_05-14-42`).

```bash
python scripts/play_rl.py --task Go2Play --num_envs 32 -r <run_folder_name> --real-time
```

After rl training, export the policies and value functions with the below, replacing `<run_folder_name>` with the name of the folder from training that is found in `checkpoints/rl/` (for example: `2025-03-14_05-14-42`).

```bash
python scripts/export_policies.py -r <run_folder_name>
```

### Data Generation

Generate data with the below, replacing `<run_folder_name>` with the name of the folder from training that is found in `checkpoints/rl/` (for example: `2025-03-14_05-14-42`). This will save data to `datasets/sim/`. The configuration for data generation is found in [`config/generate_data.yaml`](config/generate_data.yaml).

```bash
python scripts/generate_data.py rl_run=<run_folder_name>
```

Next, post-process the data with the below, replacing `<dataset_name>` with the name of the dataset in `datasets/sim/`. This script concatenates all of the episodes into single dataset files and computes parameters for input/output scaling. The configuration for data processing is found in [`config/process_data.yaml`](config/process_data.yaml). The processed dataset is saved to `datasets/sim/<dataset_name>/processed_data_{system}_{pred_len}_{hist_len}`.

```bash
python scripts/process_data.py dataset_name=<dataset_name>
```

### World Model Pretraining

Pretrain the world model with the below, replacing `<dataset_name>` with the name of the dataset in `datasets/sim/`. Checkpoints will be saved in `checkpoints/models/`; use `<run_name>` to specify the name of the checkpoints (optional). Optionally, launch with `wandb.log=true` and `wandb.entity=<your_wandb_entity_or_username>` to enable Weights and Biases logging. Additional configuration is found in [`config/train_model.yaml`](config/train_model.yaml).

```bash
python scripts/train_model.py data.dataset_name=<dataset_name> run_name=<run_name>
```

### Deployment (Simulation)

Deploy, in simulation, the pretrained world model using sampling-based planning with the below, replacing `<checkpoint_name>` with the name of the checkpoint in `checkpoints/models/`. Use `--headless` to run without a GUI. Additional configuration is found in [`config/simulate_go2.yaml`](config/simulate_go2.yaml).

```bash
python scripts/simulate_go2.py model.checkpoint=<checkpoint_name>
```