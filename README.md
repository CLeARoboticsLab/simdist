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

Pretrain the world model with the below, replacing `<dataset_name>` with the name of the dataset in `~/simdist/datasets/sim/`. Checkpoints will be saved in `~/simdist/checkpoints/models/`; use `<run_name>` to specify the name of the checkpoints (optional). Optionally, launch with `wandb.log=true` and `wandb.entity=<your_wandb_entity_or_username>` to enable Weights and Biases logging. Additional configuration is found in [`config/train_model.yaml`](config/train_model.yaml).

```bash
python scripts/train_model.py data.dataset_name=<dataset_name> run_name=<run_name>
```

## Deployment (Simulation)

Deploy, in simulation, the pretrained world model using sampling-based planning with the below, replacing `<checkpoint_name>` with the name of the checkpoint in `~/simdist/checkpoints/models/`. Use `--headless` to run without a GUI. Additional configuration is found in [`config/simulate_go2.yaml`](config/simulate_go2.yaml).

```bash
python scripts/simulate_go2.py model.checkpoint=<checkpoint_name>
```

## Deployment (Hardware)

### Hardware Setup (Go2)

Perform the following steps on the computer running the robot, which is connected to the robot via Ethernet.

First, install docker and the nvidia container toolkit, if not already installed:

```bash
sudo apt-get update
./go2_ros2_ws/setup/docker_install.sh && ./go2_ros2_ws/setup/nvidia-container-toolkit.sh
```

Next, build the container:

```bash
./go2_ros2_ws/docker/build.sh
```

List network interfaces and copy the one that is connected to the robot:

```bash
ip -o link show | awk -F': ' '{print $2}'
```

Set the interface:

```bash
cp go2_ros2_ws/.env.example go2_ros2_ws/.env
# edit go2_ros2_ws/.env and set CYCLONEDDS_IFACE=<your_interface_name>
```

### Simulation Setup (Optional)

It is possible to run all the ros2 nodes with a simulated Go2 in IsaacSim using the [go2_isaac_ros2](https://github.com/CLeARoboticsLab/go2_isaac_ros2) package. Follow the instructions in the repository to set it up.

### Startup

Start the container:

```bash
./go2_ros2_ws/scripts/run.sh
```

Inside the container, run bringup. Use the `--mocap` flag to use motion capture instead of onboard localization, and use the `--sim` flag if simulating the Go2 in IsaacSim instead of running on hardware (see [Simulation Setup (Optional)](#simulation-setup-optional)).

```bash
./scripts/bringup.sh
```
