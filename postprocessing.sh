#!/bin/bash

: '
  FLAMEGPU2 Agent-Based Model

  postprocessing.sh

  Post-process the results.

  Inputs:
      -expdir or --experiment_dir:  directory with the scenario to simulate

  Authors: Daniele Baccega, Irene Terrone, Simone Pernice
'

# Default values for input parameters
EXPERIMENT_DIR="Scenario_$(date +%s)"

while [[ $# -gt 0 ]]; do
  case $1 in
    -expdir|--experiment_dir)
      EXPERIMENT_DIR="$2"
      shift
      shift
      ;;
    -h|--help)
  	  printf "./run.sh - run the ABM\n\n"
  	  printf "Arguments:\n"
      printf "        -expdir or --experiment_dir:  directory with the scenario to simulate (default: .)\n"
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

cd results/$EXPERIMENT_DIR || { echo "Directory not found!"; exit 1; }

# Pre-check which "seed" directories exist and store them in an environment variable
existing_dirs=""
for dir in seed*; do
    if [ -d "$dir" ]; then
        # Remove spaces from the directory name before storing
        dir_no_spaces=$(echo "$dir" | tr -d '[:space:]')
        existing_dirs+="$dir_no_spaces "
    fi
done

# Loop through each .csv file in the current directory
for input_file in *.csv; do
    # Check if the file exists
    if [ -f "$input_file" ]; then
        echo "Processing $input_file..."

        # Use awk, passing the list of existing directories (without spaces)
        awk -v existing_dirs="$existing_dirs" -F, '
        BEGIN {
            # Read the single seed value from seed.txt
            getline seed_offset < "seed.txt"
            seed_offset += 0  # Ensure it is treated as a number
            close("seed.txt")

            # Parse existing_dirs into an array for quick lookup
            split(existing_dirs, dirs, " ")
            for (i in dirs) {
                dir_exists[dirs[i]] = 1
            }

            # Define mapping for col1 values
            mapping[0] = "AGENT_POSITION_AND_STATUS"
            mapping[1] = "CONTACT"
            mapping[2] = "CONTACTS_MATRIX"
            mapping[3] = "AEROSOL"
            mapping[4] = "INFO"
            mapping[5] = "DEBUG"
        }
        {
            # Extract columns
            col1 = $1
            col2 = $2

            # Add the seed value from seed.txt to col2
            col2 = col2 + seed_offset

            # Ensure col1 is within the expected range
            if (!(col1 in mapping)) {
                print "Unexpected value in col1: " col1 ". Skipping..."
                next
            }

            # Map col1 value to its corresponding label
            mapped_col1 = mapping[col1]
            gsub(/\[|\]/, "", mapped_col1)

            # Construct the directory name from col2 and remove any spaces
            dir_name = "seed" col2
            dir_name_no_spaces = gensub(/[[:space:]]/, "", "g", dir_name)

            # Check if the directory exists in the pre-checked list
            if (dir_name_no_spaces in dir_exists) {
                # Concatenate columns 3 onward into a single line
                line = ""
                for (i = 3; i <= NF; i++) {
                    line = (i == 3 ? $i : line "," $i)
                }

                # Construct the output file path
                out_file = dir_name_no_spaces "/" mapped_col1 ".csv"
                # Write the processed line to the file
                print line >> out_file
            } else {
                print "Directory " dir_name_no_spaces " does not exist. Skipping..."
            }
        }' "$input_file"
    else
        echo "No CSV files found in the directory."
    fi
done

# rm simulation.csv

cd ../../resources

# python postprocessing.py -experiment_dirs $EXPERIMENT_DIR
# python barplot.py -experiment_dirs AlarmSurgical20FromDay4 AlarmSurgical40FromDay4 AlarmSurgical80FromDay4 AlarmFFP220FromDay4 AlarmFFP240FromDay4 AlarmFFP280FromDay4 \
#                   -experiment_labels Surgical-20% Surgical-40% Surgical-80% FFP2-20% FFP2-40% FFP2-80% \
#                   -day_x 30 \
#                   -baseline_experiment AlarmNoCountermeasures
