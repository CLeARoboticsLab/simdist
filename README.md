# Simulation Distillation

This project implements **Simulation Distillation (SimDist)**, a scalable framework the distills structural priors from a simulator into a latent world model and enables rapid real-world adaptation via online planning and supervised dynamics finetuning.

[![Website](docs/badges/badge-website.svg)](https://sim-dist.github.io/)
[![Paper](docs/badges/badge-pdf.svg)](#)

## Installation

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

## Pretraining (Go2)

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

## Deployment (Real-World Go2)

### Hardware Setup

Perform the following steps on the computer running the robot, which is connected to the robot via Ethernet.

First, install docker and the nvidia container toolkit, if not already installed:

```bash
sudo apt-get update
./go2_ros2_ws/setup/docker_install.sh && ./go2_ros2_ws/setup/nvidia-container-toolkit.sh
```

Next, build the docker container:

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

If motion capture will be used (such as Vicon or OptiTrack) for localization, set the IP address of the associated VRPN server. This is usually the IP address of the computer running the motion capture software. Do this by editing the `go2_ros2_ws/.env` file and setting `MOCAP_IP=<your_mocap_ip_address>`. Set the name of the tracker to `go2` in the motion capture software. Otherwise, localization will be accomplished with LIO.

Start the docker container:

```bash
./go2_ros2_ws/scripts/run.sh
```

Build the workspace:

```bash
colcon build
```

Configure the controller parameters in [`go2_ros2_ws/src/config/config/simdist_controller.yaml`](go2_ros2_ws/src/config/config/simdist_controller.yaml), specifically the `model` and `logging` sections, to specify which checkpoint to use and if and where to log real-world data. You may also configure the control task in [`go2_ros2_ws/src/config/config/control.yaml`](go2_ros2_ws/src/config/config/control.yaml). Any time you modify any of these configuration files, you must either rebuild with `colcon build` or restart the docker container (by exiting and restarting with `./go2_ros2_ws/scripts/run.sh`).

### Simulation Setup (Optional)

It is possible to run all the ros2 nodes with a simulated Go2 in IsaacSim using the [go2_isaac_ros2](https://github.com/CLeARoboticsLab/go2_isaac_ros2) package. Follow the instructions in the repository to set it up and run the simulation.

### Startup

Start the container. Use the `--sim` flag if simulating the Go2 in IsaacSim instead of running on hardware (see [Simulation Setup (Optional)](#simulation-setup-optional)).

```bash
./go2_ros2_ws/scripts/run.sh
```

Inside the container, run bringup. Use the `--mocap` flag to use motion capture instead of onboard localization.

```bash
./scripts/bringup.sh
```

Press the `start` button on the Unitree Go2 controller **twice** to stand the robot up. (If running with `--sim`, press the space bar **twice** in the xterm window). Then, start SLAM and other support nodes with the below command; one of the tmux panes should already be populated with the command. We initialize these nodes with the robot standing up to reference the world frame from this pose.

```bash
ros2 launch bringup launch_go2.py
```

Next, start the controller with the below command; one of the tmux panes should already be populated with the command. Within the second file, you can specify if and where to log the real-world data. If you modify either of these files, you need to stop these conrol nodes, `colcon build`, and restart with the below command.

```bash
ros2 launch controller launch_go2.py
```

### Running the Robot

While the robot is standing, press the `start` button on the Unitree Go2 controller to start walking. (If running with `--sim`, use the space bar in the xterm window). Press the `start` button again while to robot is walking to make the robot stand again. Press any other button while the robot is walking to force it into a fall recovery mode. While the robot is standing or in recovery mode, press a button other than the `start` button to make the robot lie down.

### Shutdown

Detach from the tmux session with `Ctrl + b`, then `d`. Stop the container with `exit`.

## Dynamics Fintuning

### Processing Real-World Data

First, aggreage real wold data with the following, replacing `<dataset_name>` with the name of the dataset in `datasets/real`:

```bash
python scripts/aggregate_realworld_data.py dataset_name=<dataset_name>
```

Next, process the data the same way as in [Data Generation](#data-generation), with:

```bash
python scripts/process_data.py dataset_name=<dataset_name>
```

### Finetuning the World Model

Finetune the dynamics only of the world model with the below, replacing `<dataset_name>` with the name of the dataset in `datasets/real/` and `<resume_checkpoint>` with the name of the checkpoint to finetune from `checkpoints/models/`. Checkpoints will be saved in `checkpoints/models/`; use `<run_name>` to specify the name of the checkpoints (optional; it's recommended to use a different name than the original checkpoint). Optionally, launch with `wandb.log=true` and `wandb.entity=<your_wandb_entity_or_username>` to enable Weights and Biases logging. Additional configuration is found in [`config/finetune_model.yaml`](config/finetune_model.yaml).

```bash
python scripts/finetune_model.py data.dataset_name=<real_dataset> checkpoint.resume_checkpoint=<resume_checkpoint> run_name=<run_name>
```

## Acknowledgements

For SLAM, We use the version of ["point_lio_unilidar"](https://github.com/unitreerobotics/point_lio_unilidar) from [`autonomy_stack_go2`](https://github.com/jizhang-cmu/autonomy_stack_go2) from @jizhang-cmu.

## Citation

Please cite our papers:

```bibtex
@InProceedings{levy2026simdist,
  title={Simulation Distillation: Pretraining World Models in Simulation for Rapid Real-World Adaptation},
  author={Jacob Levy and Tyler Westenbroek and Kevin Huang and Fernando Palafox and Patrick Yin and Shayegan Omidshafiei and Dong-Ki Kim and Abhishek Gupta and David Fridovich-Keil},
  year={2026}
}
