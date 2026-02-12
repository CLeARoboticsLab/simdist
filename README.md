# Simulation Distillation

## Setup

Create a new conda environment [optional, but recommended]:

```bash
conda create -n simdist python=3.10
conda activate simdist
```

Install Isaac Sim:

```bash
pip install --upgrade pip
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
```

Clone the repo and install IsaacLab:

```bash
sudo apt install cmake build-essential
cd
git clone --recurse-submodules https://github.com/CLeARoboticsLab/simdist.git
cd simdist
./IsaacLab/isaaclab.sh -i none
```

Install simdist:

```bash
pip install -e .
```

## Expert Policy Training

To train, use the below, replacing `<wandb_username>` with your Weights and Biases username (if you don't have an account, create one at [https://wandb.ai/](https://wandb.ai/)). Runs are saved to `~/simdist/checkpoints/rl/`. The environment config is found in [`simdist/rl/go2.py`](simdist/rl/go2.py) and the PPO config is found in [`simdist/rl/rsl_rl_ppo_cfg.py`](simdist/rl/rsl_rl_ppo_cfg.py).

```bash
WANDB_USERNAME=<wandb_username> python scripts/train_rl.py --task Go2 --headless
```

To visualize and play an expert policy, use the below, replacing `<run_folder_name>` with the name of the folder from training that is found in `~/simdist/checkpoints/rl/` (for example: `2025-03-14_05-14-42`).

```bash
python scripts/play_rl.py --task Go2Play --num_envs 32 -r <run_folder_name> --real-time
```

After rl training, export the policies and value functions with the below, replacing `<run_folder_name>` with the name of the folder from training that is found in `~/simdist/checkpoints/rl/` (for example: `2025-03-14_05-14-42`).

```bash
python scripts/export_policies.py -r <run_folder_name>
```

## Data Generation

Generate data with the below, replacing `<run_folder_name>` with the name of the folder from training that is found in `~/simdist/checkpoints/rl/` (for example: `2025-03-14_05-14-42`). This will save data to `~/simdist/datasets/sim/`. The configuration for data generation is found in [`config/generate_data.yaml`](config/generate_data.yaml).

```bash
python scripts/generate_data.py rl_run=<run_folder_name>
```

Next, post-process the data with the below, replacing `<dataset_name>` with the name of the dataset in `~/simdist/datasets/sim/`. This script concatenates all of the episodes into single dataset files and computes parameters for input/output scaling. The configuration for data processing is found in [`config/process_data.yaml`](config/process_data.yaml). The processed dataset is saved to `~/simdist/datasets/sim/<dataset_name>/processed_data_{system}_{pred_len}_{hist_len}`.

```bash
python scripts/process_data.py dataset_name=<dataset_name>
```

## World Model Pretraining

Pretrain the world model with the below, replacing `<dataset_name>` with the name of the dataset in `~/simdist/datasets/sim/`. Checkpoints will be saved in `~/simdist/checkpoints/rl/`; use `<run_name>` to specify the name of the checkpoints (optional). Optionally, launch with `wandb.log=true` and `wandb.entity=<your_wandb_entity_or_username>` to enable Weights and Biases logging.

```bash
python scripts/train_model.py data.dataset_name=<dataset_name> run_name=<run_name>
```
