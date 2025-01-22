#!/bin/bash

# Comparison between NetLogo and FLAME GPU 2
cd NetLogoSchoolModel
./run.sh Configurations/F4FComparison/NoCountermeasures.conf 16
./run.sh Configurations/F4FComparison/Countermeasures.conf 16
cd ..

./abm_ensemble -expdir NoCountermeasures -prun 16 -c ON
./abm_ensemble -expdir Countermeasures -prun 16 -c ON

