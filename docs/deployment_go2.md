## Deployment (Real-World Go2)

- [Hardware Setup](#hardware-setup)
- [Simulation Setup (Optional)](#simulation-setup-optional)
- [Startup](#startup)
- [Running the Robot](#running-the-robot)
- [Shutdown](#shutdown)

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