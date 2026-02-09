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
./IsaacLab/isaaclab.sh -i rsl_rl
```

Install simdist:

```bash
pip install -e .
```

## Expert Policy Training

To train, use the below. Runs are saved to `~/simdist/checkpoints/rl/`. The environment config is found in `~/simdist/simdist/rl/go2.py` and the PPO config is found in `~~/simdist/simdist/rl/rsl_rl_ppo_cfg.py`.

```bash
python scripts/train_rl.py --task Go2 --headless
```

To visualize and play an expert policy, use the below, replacing `<run_folder_name>` with the name of the folder from training that is found in `~/simdist/checkpoints/rl/` (for example: `2025-03-14_05-14-42`).

```bash
python scripts/play_rl.py --task Go2Play --num_envs 32 -r <run_folder_name> --real-time
```
