#!/bin/bash

: '
  FLAMEGPU2 Agent-Based Model

  abm.sh

  Generate the configuration file, build and run the ABM.

  Inputs:
      -expdir or --experiment_dir:         directory with the scenario to simulate
      -ob     or --only_build:             build the model without execute it
   	  -v      or --visualisation:          activate the visualisation
      -cps    or --checkpoint_simulation:  run the model in a simplified version with the aim to obtain a checkpoint
      -g      or --debug:                  execute the simulation with debug prints
      -c      or --clean:                  clean old files and directories

  Authors: Daniele Baccega, Irene Terrone, Simone Pernice
'

bash drivers_check.sh
if [ $? -eq 1 ];
then
  exit 1
fi

# Default values for input parameters
EXPERIMENT_DIR="None"
RESULTS_DIR="results"
ONLY_BUILD="OFF"
VISUALISATION="OFF"
CHECKPOINT_SIMULATION="OFF"
DEBUG="OFF"
CLEAN="OFF"
SUBSTITUTE_DIR="OFF"


while [[ $# -gt 0 ]]; do
  case $1 in
    -expdir|--experiment_dir)
      EXPERIMENT_DIR="$2"
      shift
      shift
      ;;
    -ob|--only_build)
      ONLY_BUILD="$2"
      shift
      shift
      ;;
    -v|--visualisation)
      VISUALISATION="$2"
      shift
      shift
      ;;
    -g|--debug)
      DEBUG="$2"
      shift
      shift
      ;;
    -c|--clean)
      CLEAN="$2"
      shift
      shift
      ;;
    -subdir|--substitute_dir)
      SUBSTITUTE_DIR="$2"
      shift
      shift
      ;;
    -h|--help)
  	  printf "./run.sh - run the ABM\n\n"
  	  printf "Arguments:\n"
      printf "        -expdir or --experiment_dir:         directory with the scenario to simulate\n"
      printf "        -ob     or --only_build:             build the model without execute it (default: OFF; possible values: ON, OFF)\n"
      printf "        -v      or --visualisation:          activate the visualisation (default: OFF; possible values: ON, OFF)\n"
      printf "        -g      or --debug:                  execute the simulation with debug prints (default: OFF; possible values: ON, OFF)\n"
      printf "        -c      or --clean:                  clean old files and directories (default: OFF; possible values: ON, OFF)\n"
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

DIR_PATH="results/$EXPERIMENT_DIR"

if [ -d "$DIR_PATH" ]; then
    read -p "The directory '$DIR_PATH' already exists. Do you want to replace it? (y/n): " response
    if [ "$response" == "y" ] || [ $SUBSTITUTE_DIR == "ON" ]; then
        rm -rf "$DIR_PATH"  # Remove the directory and its contents
        mkdir -p "$DIR_PATH"  # Recreate the directory
        echo "The directory '$DIR_PATH' has been replaced."
    else
        echo "Operation canceled. The existing directory was not replaced."
        exit 1
    fi
else
    mkdir -p "$DIR_PATH"  # Create the directory if it doesn't exist
    echo "The directory '$DIR_PATH' has been created."
fi

if [ ! -d flamegpu2 ];
then
  python3 -m venv flamegpu2
  source flamegpu2/bin/activate
  pip install -r flamegpu2-python.txt
else
  source flamegpu2/bin/activate
fi

if [ $CLEAN == "ON" ];
then
  bash clean.sh
fi

# Generate the configuration file to give in input to the ABM model
WHOLE_OUTPUT="$(bash generate_configuration.sh -e OFF -expdir $EXPERIMENT_DIR 2>&1)"
#bash generate_configuration.sh -e OFF -expdir $EXPERIMENT_DIR 2>&1
SEED="$(echo "$WHOLE_OUTPUT" | cut -d' ' -f1)"

echo $SEED > results/$EXPERIMENT_DIR/seed.txt

# Build the model
bash build.sh -cps $CHECKPOINT_SIMULATION -g $DEBUG -v $VISUALISATION

if [ $ONLY_BUILD == "OFF" ];
then
  # Run the model
  bash run.sh -expdir $EXPERIMENT_DIR -v $VISUALISATION -e OFF
fi

if [ -f /.dockerenv ]; then
  cp -r results flamegpu2_results
  chmod -R 777 flamegpu2_results/results
else
  if [ "$RESULTS_DIR" != "results" ]; then
    cp -r results $RESULTS_DIR
    chmod -R 777 $RESULTS_DIR/results
  fi
fi

deactivate
