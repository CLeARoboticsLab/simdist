#!/usr/bin/env bash
set -euo pipefail

# ROS setup scripts can read unset vars (e.g., AMENT_TRACE_SETUP_FILES),
# so source them with nounset temporarily disabled.
source_relaxed_nounset() {
    local script_path="$1"
    set +u
    # shellcheck disable=SC1090
    source "${script_path}"
    set -u
}

# Load ROS environment so ros2 commands and Python packages resolve correctly.
if [ -n "${ROS_DISTRO:-}" ] && [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
    source_relaxed_nounset "/opt/ros/${ROS_DISTRO}/setup.bash"
fi

# If a workspace overlay is available, source it too.
if [ -n "${ROS2_WS:-}" ] && [ -f "${ROS2_WS}/install/setup.bash" ]; then
    source_relaxed_nounset "${ROS2_WS}/install/setup.bash"
elif [ -n "${ROS_WS:-}" ] && [ -f "${ROS_WS}/install/setup.bash" ]; then
    source_relaxed_nounset "${ROS_WS}/install/setup.bash"
fi

if [ "${SKIP_SIMDIST_EDITABLE_INSTALL:-0}" != "1" ]; then
    if [ -z "${SIMDIST_WORKSPACE_ROOT:-}" ]; then
        echo "Error: SIMDIST_WORKSPACE_ROOT is not set." >&2
        exit 1
    fi

    if [ ! -f "${SIMDIST_WORKSPACE_ROOT}/pyproject.toml" ]; then
        echo "Error: ${SIMDIST_WORKSPACE_ROOT}/pyproject.toml not found." >&2
        exit 1
    fi

    python -m pip install --no-deps --no-build-isolation -e "${SIMDIST_WORKSPACE_ROOT}"
fi

exec "$@"
