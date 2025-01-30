#!/bin/bash

# Comparison between NetLogo and FLAME GPU 2 (ITC test, 200 runs)
cd NetLogoSchoolModel
./run.sh Configurations/F4FComparison/NoCountermeasures.conf 16
./run.sh Configurations/F4FComparison/Countermeasures.conf 16
cd ..

./run_docker_ensemble NoCountermeasures 16
./run_docker_ensemble Countermeasures 16


