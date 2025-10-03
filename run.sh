#!/bin/bash

: '
  FLAMEGPU 2 Agent-Based Model

  run.sh

  Run the ABM.

  Inputs:
      -expdir or --experiment_dir:   directory with the scenario to simulate
      -prun   or --parallel_run:     number of run to execute in parallel on GPUs
      -e      or --ensemble:         run with ensemble
   	  -v      or --visualisation:    activate the visualisation

  Authors: Daniele Baccega, Irene Terrone, Simone Pernice
'

# Default values for input parameters
EXPERIMENT_DIR="None"
PARALLEL_RUN="10"
ENSEMBLE="OFF"
VISUALISATION="OFF"

while [[ $# -gt 0 ]]; do
  case $1 in
    -expdir|--experiment_dir)
      EXPERIMENT_DIR="$2"
      shift
      shift
      ;;
    -prun|--parallel_run)
      PARALLEL_RUN="$2"
      shift
      shift
      ;;
    -e|--ensemble)
      ENSEMBLE="$2"
      shift
      shift
      ;;
    -v|--visualisation)
      VISUALISATION="$2"
      shift
      shift
      ;;
    -h|--help)
  	  printf "./run.sh - run the ABM\n\n"
  	  printf "Arguments:\n"
      printf "        -expdir or --experiment_dir:   directory with the scenario to simulate"
      printf "        -prun   or --parallel_run:     number of run to execute in parallel on GPUs (default: 10)\n"
      printf "        -e      or --ensemble:         run with ensemble (default: OFF; possible values: ON, OFF)\n"
      printf "        -v      or --visualisation:    activate the visualisation (default: OFF; possible values: ON, OFF)\n"
      exit 1
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1")   # Save positional arg
      shift                     # Past argument
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}"  # Restore positional parameters

DIR="resources/f4f/$EXPERIMENT_DIR"

# Check if the directory exists
if [ ! -d "$DIR" ]; then
    echo "❌ Error: Directory $DIR does not exist."
    exit 1
fi

if [ $VISUALISATION != "ON" ];
then
  VISUALISATION_ARG="-c"
fi

if [ $ENSEMBLE == "ON" ];
then
  __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./build/bin/Release/FLAMEGPUABM -c $PARALLEL_RUN $EXPERIMENT_DIR
else
  __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./build/bin/Release/FLAMEGPUABM -i resources/configuration_file.xml $EXPERIMENT_DIR
fi

bash postprocessing.sh -expdir $EXPERIMENT_DIR
