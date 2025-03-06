#!/bin/bash

# Comparison between NetLogo and FLAME GPU 2 (ITC test, 200 runs)
# cd NetLogoSchoolModel
# ./run.sh Configurations/F4FComparison/NoCountermeasures.conf 16
# ./run.sh Configurations/F4FComparison/Countermeasures.conf 16
# cd ..

# ./run_docker_ensemble.sh SchoolComparisonNoCountermeasures
./run_docker_ensemble.sh SchoolComparisonCountermeasures

cd resources
R -e "if (!requireNamespace('dplyr', quietly = TRUE)) install.packages('dplyr')"
R -e "if (!requireNamespace('fdatest', quietly = TRUE)) install.packages('fdatest')"
R -e "if (!requireNamespace('patchwork', quietly = TRUE)) install.packages('patchwork')"
R -e "if (!requireNamespace('ggplot2', quietly = TRUE)) install.packages('ggplot2')"

cd FLAMEGPUvsNetLogo
Rscript ITP.R
cd ../..