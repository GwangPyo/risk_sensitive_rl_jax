FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

# declare the image name
ENV IMG_NAME=jax/jax-gpu
    # declare what jaxlib tag to use
    # if a CI/CD system is expected to pass in these arguments
    # the dockerfile should be modified accordingly

# install python3-pip
RUN apt update && apt install python3-pip -y

# install dependencies via pip
RUN pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN apt install git -y

RUN pip install dm-haiku
RUN pip install optax
RUN pip install pandas
RUN pip install git+https://github.com/GwangPyo/risk_sensitive_rl_jax


ENV XLA_PYTHON_CLIENT_PREALLOCATE false

# install gym box2d
RUN apt install swig -y
RUN pip install gym[box2d]

# install mujoco

ARG DEBIAN_FRONTEND=noninteractive
RUN apt install wget -y &&   apt install libglfw3 -y &&  apt install libglfw3-dev -y

RUN apt install  libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev\
    patchelf -y

RUN pip install imageio
RUN pip install mujoco-py
RUN pip install mujoco==2.2.0

RUN mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz

ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
# compile
RUN echo "import gym; gym.make('Hopper-v3'); print('compiled')" | python3
RUN apt-get clean