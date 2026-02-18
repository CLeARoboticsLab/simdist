cd $ROS2_WS
colcon build

USE_SIM="${USE_SIM:-false}"
MOCAP=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mocap|mocap=true)
            MOCAP=true
            shift
            ;;
        *)
            echo "Warning: unknown argument '$1' ignored."
            shift
            ;;
    esac
done

echo "Using simulation: $USE_SIM"
echo "Using mocap: $MOCAP"

USE_SIM="$USE_SIM" MOCAP="$MOCAP" tmuxp load "$ROS2_WS/tmuxp.yaml"
