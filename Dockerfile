# Base image https://hub.docker.com/u/rocker/
FROM nvcr.io/nvidia/cuda:12.2.0-devel-ubuntu22.04
LABEL maintainer="Daniele Baccega <daniele.baccega@unito.it>"

RUN apt update \
    && apt install -y build-essential cmake doxygen git unzip pciutils gawk r-base \
    && apt install -y wget nano \
    && apt install -y libglu1-mesa-dev freeglut3-dev mesa-common-dev libxmu-dev libxi-dev libgl-dev libfreetype6-dev libfontconfig1-dev libdevil-dev \
    && apt install -y python3 python3-pip python3-venv \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

RUN R -e "if (!requireNamespace('dplyr', quietly = TRUE)) install.packages('dplyr')"
RUN R -e "if (!requireNamespace('fdatest', quietly = TRUE)) install.packages('fdatest')"
RUN R -e "if (!requireNamespace('patchwork', quietly = TRUE)) install.packages('patchwork')"
RUN R -e "if (!requireNamespace('ggplot2', quietly = TRUE)) install.packages('ggplot2')"

RUN mkdir -p /home/docker/flamegpu2

RUN git clone https://github.com/qBioTurin/FLAMEGPU-FORGE4FLAME.git /home/docker/flamegpu2/FLAMEGPU-FORGE4FLAME

RUN chmod -R 777 /home/docker/flamegpu2

WORKDIR /home/docker/flamegpu2/FLAMEGPU-FORGE4FLAME