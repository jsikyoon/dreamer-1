#from tensorflow/tensorflow:1.13.1-gpu-py3
from tensorflow/tensorflow:1.15.2-gpu-py3
#from nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# basic libraries
RUN apt-get update && apt-get install -y \
    curl \
    zip unzip \
    python3-dev \
    python3-pip \
    git

#######################Jsik option##########################

RUN apt-get install -y \
    vim \
    net-tools \
    openssh-server \
    screen

# Setting
RUN git clone https://github.com/jsikyoon/my_ubuntu_settings --branch indent2 && \
    cd my_ubuntu_settings && \
    ./setup.sh && \
    rm -rf /my_ubuntu_settings

############################################################

#######################Dmlab option##########################

RUN apt install -y libgl1-mesa-glx ffmpeg

RUN pip3 install \
    #tensorflow-gpu==1.13.1 \
    #tensorflow_probability==0.6.0 \
    tensorflow_probability==0.8.0 \
    dm_control pandas matplotlib \
    gym gym[atari] ruamel.yaml \
    scikit-image

#############################################################

#######################Dmlab option##########################

RUN apt-get update && apt-get install -y \
    curl \
    zip \
    unzip \
    software-properties-common \
    pkg-config \
    g++-4.8 \
    zlib1g-dev \
    lua5.1 \
    liblua5.1-0-dev \
    libffi-dev \
    gettext \
    freeglut3 \
    libsdl2-dev \
    libosmesa6-dev \
    libglu1-mesa \
    libglu1-mesa-dev \
    python3-dev \
    build-essential \
    git \
    python-setuptools \
    python3-pip \
    libjpeg-dev \
    tmux

RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | \
    tee /etc/apt/sources.list.d/bazel.list && \
    curl https://bazel.build/bazel-release.pub.gpg | \
    apt-key add - && \
    apt-get update && apt-get install -y bazel

# Build and install DeepMind Lab pip package.
# We explicitly set the Numpy path as shown here:
# https://github.com/deepmind/lab/blob/master/docs/users/build.md
RUN NP_INC="$(python3 -c 'import numpy as np; print(np.get_include()[5:])')" && \
    #git clone https://github.com/deepmind/lab.git && \
    git clone https://github.com/jsikyoon/lab.git --branch jsik/gr_seed_rl && \
    cd lab && \
    #git checkout 937d53eecf7b46fbfc56c62e8fc2257862b907f2 && \
    sed -i 's@python3.5@python3.6@g' python.BUILD && \
    sed -i 's@glob(\[@glob(["'"$NP_INC"'/\*\*/*.h", @g' python.BUILD && \
    sed -i 's@: \[@: ["'"$NP_INC"'", @g' python.BUILD

RUN cd lab && \
    bazel build -c opt python/pip_package:build_pip_package --incompatible_remove_legacy_whole_archive=0 && \
    pip3 install wheel && \
    PYTHON_BIN_PATH="/usr/bin/python3" ./bazel-bin/python/pip_package/build_pip_package /tmp/dmlab_pkg && \
    pip3 install /tmp/dmlab_pkg/DeepMind_Lab-*.whl --force-reinstall && \
    rm -rf /lab

# Install dataset (from https://github.com/deepmind/lab/tree/master/data/brady_konkle_oliva2008)
RUN mkdir dataset && \
    cd dataset && \
    apt install -y python-pip && \
    pip install Pillow && \
    pip3 install Pillow && \
    curl -sS https://raw.githubusercontent.com/deepmind/lab/master/data/brady_konkle_oliva2008/README.md | \
    tr '\n' '\r' | \
    sed -e 's/.*```sh\(.*\)```.*/\1/' | \
    tr '\r' '\n' | \
    bash

#############################################################

# Test
RUN git clone https://github.com/jsikyoon/dreamer-1 && \
    cd dreamer-1 && \
    python3 -m dreamer.scripts.train --logdir ./logdir/debug --params '{defaults: [dreamer, debug], tasks: [dummy]}' \
        --num_runs 1 --resume_runs False
