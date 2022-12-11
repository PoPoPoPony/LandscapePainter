# FROM nvidia/cuda:11.7.0-base-ubuntu20.04
FROM amoxley/stylegan2-ada-pytorch:cuda11.1-2

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

# RUN apt update
# RUN apt install software-properties-common
# RUN add-apt-repository ppa:deadsnakes/ppa
# RUN apt-get update
# RUN apt-get install -y python3.9
# RUN apt install -y python3-pip
# RUN apt-get -y install python3-pip

# COPY ./cpp_extension.py ../opt/conda/lib/python3.9/site-packages/torch/utils/
RUN pip install --upgrade -r /code/requirements.txt
# RUN python setup.py clean
# RUN pip install torch torchvision torchaudio
# ENV DEBIAN_FRONTEND noninteractive


# RUN apt-get update -y
# RUN apt-get install -y build-essential

# RUN apt-get update -y && \
#     apt-get -y install gcc mono-mcs && \
#     rm -rf /var/lib/apt/lists/*

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     pkg-config \
#     libglvnd0 \
#     libgl1 \
#     libglx0 \
#     libegl1 \
#     libgles2 \
#     libglvnd-dev \
#     libgl1-mesa-dev \
#     libegl1-mesa-dev \
#     libgles2-mesa-dev \
#     cmake \
#     curl \
#     libsm6 \
#     libxext6 \
#     libxrender-dev


# RUN apt-get -y install libgl-dev
# RUN apt-get install ninja-build
# opt/conda/lib/python3.9/site-packages/Ninja

# RUN apt install -y wget
# RUN wget https://github.com/ninja-build/ninja/releases/download/v1.10.2/ninja-linux.zip
# RUN apt-get install -y unzip
# RUN unzip ninja-linux.zip -d /usr/local/bin/
# RUN update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force


COPY ./app /code/app
# CMD bin/sh


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8084"]
# ENTRYPOINT /bin/bash

