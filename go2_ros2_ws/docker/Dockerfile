FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV LANG=en_US.UTF-8
ENV LC_ALL=C.UTF-8 
ENV DEBIAN_FRONTEND=noninteractive
ARG TZ=Etc/UTC
ENV TZ=${TZ}
RUN echo '${TZ}' > /etc/timezone && \
    ln -sf /usr/share/zoneinfo/${TZ} /etc/localtime
ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV QT_X11_NO_MITSHM=1

# setup locales
RUN apt-get update && \
    apt-get install -q -y --no-install-recommends locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*

# install base packages
COPY base_pkgs.txt /tmp/base_pkgs.txt
RUN \
    apt-get update && \
    apt-get install -y --no-install-recommends $(cat /tmp/base_pkgs.txt) && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/base_pkgs.txt

# setup ROS
ENV ROS_DISTRO=humble
COPY ros2_install.sh /tmp/ros2_install.sh
RUN \
    bash /tmp/ros2_install.sh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/ros2_install.sh

# Create user and setup permissions on /etc/sudoers
ARG UID=1000
ARG GID=1000
ARG WORKSPACE=go2_ros2_ws
ARG DEFAULT_USER=go2
ARG CMAKE_LOCAL_FETCH=/home/${DEFAULT_USER}/endurance_ws/cmake_local_fetch
ENV CMAKE_LOCAL_FETCH=${CMAKE_LOCAL_FETCH}
ENV DEFAULT_USER=${DEFAULT_USER}
ENV DEFAULT_GROUP=$DEFAULT_USER
ENV DEFAULT_GID=${GID}
ENV DEFAULT_UID=${UID}
ENV HOME=/home/$DEFAULT_USER
RUN groupadd -o -g $GID $DEFAULT_USER && useradd -m -s /bin/bash -N -u $UID -g $GID -G sudo $DEFAULT_USER && \
    echo "${DEFAULT_USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && \
    chmod 0440 /etc/sudoers && \
    chmod g+w /etc/passwd  && \
    chown -R ${DEFAULT_USER}:${DEFAULT_GROUP} "${HOME}"
USER $DEFAULT_USER
ENV ROS_WS=$HOME/$WORKSPACE
RUN mkdir -p $ROS_WS
WORKDIR $HOME
ARG CMAKE_LOCAL_FETCH=${ROS_WS}/cmake_local_fetch

# Isolate Python deps from apt-managed system packages.
ENV SIMDIST_VENV=${HOME}/.simdist_venv
RUN python3 -m virtualenv --system-site-packages "${SIMDIST_VENV}"
ENV VIRTUAL_ENV="${SIMDIST_VENV}"
ENV PATH="${SIMDIST_VENV}/bin:${PATH}"

# Ensure editable install tooling supports PEP 660.
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python deps for simdist
COPY --chown=${UID}:${GID} simdist.requirements.txt ${HOME}/simdist.requirements.txt
RUN python -m pip install --no-cache-dir -r "${HOME}/simdist.requirements.txt" && \
    rm -f "${HOME}/simdist.requirements.txt"

# Install extra pip deps
COPY --chown=${UID}:${GID} pip_pkgs.txt ${HOME}/pip_pkgs.txt
RUN python -m pip install --no-cache-dir --upgrade -r "${HOME}/pip_pkgs.txt" && \
    rm -f "${HOME}/pip_pkgs.txt"

# Add configs
ENV TERM=xterm-256color
ENV PATH=${PATH}:${HOME}/.local/bin
COPY .tmux.conf /etc/tmux.conf

# Runtime startup hook: refresh editable simdist install each container start.
COPY --chmod=755 entrypoint.sh /usr/local/bin/go2_entrypoint.sh
ENTRYPOINT ["/usr/local/bin/go2_entrypoint.sh"]
CMD ["bash"]
