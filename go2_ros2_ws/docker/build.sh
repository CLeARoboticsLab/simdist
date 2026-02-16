#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYPROJECT_PATH="${REPO_ROOT}/pyproject.toml"
OUTPUT_PATH="${SCRIPT_DIR}/simdist.requirements.txt"

if [ ! -f "${PYPROJECT_PATH}" ]; then
    echo "Error: pyproject.toml not found at ${PYPROJECT_PATH}" >&2
    exit 1
fi

# Extract [project].dependencies entries from pyproject.toml.
awk '
BEGIN {
    in_project = 0
    in_dependencies = 0
}
/^[[:space:]]*\[project\][[:space:]]*$/ {
    in_project = 1
    next
}
/^[[:space:]]*\[[^]]+\][[:space:]]*$/ {
    if (in_project && in_dependencies) {
        exit
    }
    in_project = 0
}
in_project && /^[[:space:]]*dependencies[[:space:]]*=[[:space:]]*\[/ {
    in_dependencies = 1
    next
}
in_project && in_dependencies {
    if ($0 ~ /^[[:space:]]*][[:space:]]*$/) {
        in_dependencies = 0
        exit
    }

    if ($0 ~ /^[[:space:]]*"[^"]+"[[:space:]]*,?[[:space:]]*$/) {
        line = $0
        sub(/^[[:space:]]*"/, "", line)
        sub(/"[[:space:]]*,?[[:space:]]*$/, "", line)
        print line
    }
}
' "${PYPROJECT_PATH}" > "${OUTPUT_PATH}"

if [ ! -s "${OUTPUT_PATH}" ]; then
    echo "Error: failed to extract dependencies from ${PYPROJECT_PATH}" >&2
    exit 1
fi

TZ_VALUE="${TZ:-$(cat /etc/timezone 2>/dev/null || echo Etc/UTC)}"
IMAGE_NAME="${IMAGE_NAME:-go2_ros2}"
UID_VALUE="${UID:-$(id -u)}"
GID_VALUE="${GID:-$(id -g)}"

echo "# Generated ${OUTPUT_PATH} from ${PYPROJECT_PATH}"
echo "# Building Docker image '${IMAGE_NAME}' in ${SCRIPT_DIR}"
echo "# Using UID=${UID_VALUE} GID=${GID_VALUE}"

cd "${SCRIPT_DIR}"
docker build \
    --build-arg TZ="${TZ_VALUE}" \
    --build-arg UID="${UID_VALUE}" \
    --build-arg GID="${GID_VALUE}" \
    --network=host \
    -t "${IMAGE_NAME}" \
    "$@" \
    .
