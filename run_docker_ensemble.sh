#!/bin/bash

docker run --user $UID:$UID --rm --gpus all --runtime nvidia -v $(pwd):/home/docker/flamegpu2/FLAMEGPU-FORGE4FLAME/flamegpu2_results qbioturin/flamegpu2:School /usr/bin/bash -c "/home/docker/flamegpu2/FLAMEGPU-FORGE4FLAME/abm_ensemble.sh -expdir $1"