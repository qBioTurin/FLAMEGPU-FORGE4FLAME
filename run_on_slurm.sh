#!/bin/bash

# Define arrays
PARTITIONS=("MachineName1-booked", "MachineName2-booked", ..., "MachineNameN-booked")
RESERVATIONS=("Run1", "Run2", ..., "RunN")
GRES=("GPUName1", "GPUName2", ..., "GPUNameN")
MODEL_NAMES=("NameOfTheModel1", "NameOfTheModel2", ..., "NameOfTheModelN")

ARRAY_LENGTH=${#PARTITIONS[@]}

for ((i=0; i<ARRAY_LENGTH; i++)); do
    JOB_SCRIPT="job_script$i.sh"
    
    cat <<EOF > "$JOB_SCRIPT"
#!/bin/bash
#SBATCH --partition=${PARTITIONS[$i]}
#SBATCH --reservation=${RESERVATIONS[$i]}
#SBATCH -N 1
#SBATCH --gres=${GRES[$i]}

./abm_ensemble.sh -expdir ${MODEL_NAMES[$i]}
EOF

    # Submit the job and check if it was successful
    if sbatch "$JOB_SCRIPT"; then
        echo "Submitted: $JOB_SCRIPT"
    else
        echo "Failed to submit: $JOB_SCRIPT" >&2
    fi
done