# Base image https://hub.docker.com/u/rocker/
FROM nvcr.io/nvidia/cuda:12.3.0-devel-ubuntu22.04
LABEL maintainer="Daniele Baccega <daniele.baccega@unito.it>"

# Imposta una directory temporanea alternativa
ENV TMPDIR=/var/tmp
RUN mkdir -p $TMPDIR && chmod 1777 $TMPDIR

RUN apt update \
    && apt install -y build-essential cmake doxygen git unzip \
    && apt install -y wget \
    && apt install -y libglu1-mesa-dev freeglut3-dev mesa-common-dev libxmu-dev libxi-dev libgl-dev libfreetype6-dev libfontconfig1-dev libdevil-dev \
    && apt install -y python3 python3-pip python3-venv \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# SDL
# ENV TMP_SDL=${TMPDIR}/SDL
# RUN wget -nc https://github.com/libsdl-org/SDL/releases/download/release-2.30.2/SDL2-2.30.2.zip \
#     && unzip SDL2-2.30.2.zip -d $TMP_SDL \
#     && rm SDL2-2.30.2.zip
# RUN cd ${TMP_SDL}/SDL2-2.30.2 \
#     && ./configure \
#     && make \
#     && make install

# # GLM
# ENV TMP_GLM=${TMPDIR}/GLM
# RUN git clone https://github.com/g-truc/glm.git $TMP_GLM
# RUN cmake -DGLM_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=OFF -B ${TMP_GLM}/build $TMP_GLM \
#     && cmake --build ${TMP_GLM}/build -- all \
#     && cmake --build ${TMP_GLM}/build -- install

# # GLEW
# ENV TMP_GLEW=${TMPDIR}/glew
# RUN git clone https://github.com/nigels-com/glew.git $TMP_GLEW
# RUN make extensions -C ${TMP_GLEW} \
#     && make -C ${TMP_GLEW} \
#     && make install -C ${TMP_GLEW} \
#     && make clean -C ${TMP_GLEW}

# Create flamegpu2 directory
RUN mkdir -p /home/docker/flamegpu2/results && chmod -R 777 /home/docker/flamegpu2

WORKDIR /home/docker/flamegpu2/

RUN git clone https://francescosiv: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@github.com/qBioTurin/FLAMEGPU-FORGE4FLAME.git

WORKDIR /home/docker/flamegpu2/FLAMEGPU-FORGE4FLAME