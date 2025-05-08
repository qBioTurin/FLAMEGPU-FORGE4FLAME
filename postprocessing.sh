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
      printf "        -expdir or --experiment_dir:  directory with the scenario to simulate\n"
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

DIR="results/$EXPERIMENT_DIR"

# Check if the directory exists
if [ ! -d "$DIR" ]; then
    echo "❌ Error: Directory $DIR does not exist."
    exit 1
fi

cd $DIR

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
            # Read the seed value from seed.txt (a single numeric value)
            getline seed_offset < "seed.txt"
            seed_offset += 0
            close("seed.txt")

            # Build a lookup of existing directory names (assumed space-separated in existing_dirs)
            split(existing_dirs, dirs, " ")
            for (i in dirs)
                dir_exists[dirs[i]] = 1

            # Map the numeric code in column 1 to a category label
            mapping[0] = "AGENT_POSITION_AND_STATUS"
            mapping[1] = "CONTACT"
            mapping[2] = "CONTACTS_MATRIX"
            mapping[3] = "AEROSOL"
            mapping[4] = "INFO"
            mapping[5] = "DEBUG"
            mapping[6] = "TEMPORARY_DEBUG"

            # Define the expected number of columns for each category
            expected["AGENT_POSITION_AND_STATUS"] = 9
            expected["CONTACT"] = 6
            expected["CONTACTS_MATRIX"] = 6
            expected["AEROSOL"] = 5
            expected["INFO"] = 4
        }
        {
            col1 = $1
            col2 = $2

            # Only process if col1 maps to a known category
            if (!(col1 in mapping)) {
                print "Unexpected value in col: " col1 ". Skipping..."
                next
            }
            category = mapping[col1]

            # If the category has an expected column count, verify it
            if ((category in expected) && (NF != expected[category])) {
                print "Skipping line due to incorrect column count for " category ": " $0 " (" NF " fields, expected " expected[category] ")" > "/dev/stderr"
                next
            }

            # Adjust col2 by adding the seed offset
            col2 = col2 + seed_offset

            # Build the directory name from col2 (removing any spaces)
            dir_name = "seed" col2
            dir_name_no_spaces = gensub(/[[:space:]]/, "", "g", dir_name)

            # Only process if the directory exists in the pre-checked list
            if (!(dir_name_no_spaces in dir_exists)) {
                print "Directory " dir_name_no_spaces " does not exist. Skipping..." > "/dev/stderr"
                next
            }

            # Concatenate columns 3 through NF into a single comma-separated string
            line = ""
            for (i = 3; i <= NF; i++) {
                line = (i == 3 ? $i : line "," $i)
            }

            # Define the output file path and write the processed line
            out_file = dir_name_no_spaces "/" category ".csv"
            print line >> out_file
        }
    ' "$input_file"
    else
        echo "No CSV files found in the directory."
    fi
done

rm simulation.csv

cd ../..