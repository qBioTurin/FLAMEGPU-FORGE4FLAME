#!/bin/bash

./run_docker_ensemble.sh AlarmNoCountermeasures
./run_docker_ensemble.sh AlarmSurgical20
./run_docker_ensemble.sh AlarmSurgical40
./run_docker_ensemble.sh AlarmSurgical80
./run_docker_ensemble.sh AlarmFFP220
./run_docker_ensemble.sh AlarmFFP240
./run_docker_ensemble.sh AlarmFFP280

if [ ! -d flamegpu2 ];
then
  python3 -m venv flamegpu2
  source flamegpu2/bin/activate
  pip install -r flamegpu2-python.txt
else
  source flamegpu2/bin/activate
fi

cd resources
python postprocessing.py
python barplot.py -experiment_dirs AlarmNoCountermeasures AlarmSurgical20 AlarmSurgical40 AlarmSurgical80 AlarmFFP220 AlarmFFP240 AlarmFFP280 \
                  -experiment_labels Baseline Surgical-20% Surgical-40% Surgical-80% FFP2-20% FFP2-40% FFP2-80% \
                  -day_x 30 \
                  -baseline_experiment AlarmNoCountermeasures
python barplot.py -experiment_dirs AlarmNoCountermeasures AlarmSurgical20 AlarmSurgical40 AlarmSurgical80 AlarmFFP220 AlarmFFP240 AlarmFFP280 \
                  -experiment_labels Baseline Surgical-20% Surgical-40% Surgical-80% FFP2-20% FFP2-40% FFP2-80% \
                  -day_x 60 \
                  -baseline_experiment AlarmNoCountermeasures
python barplot.py -experiment_dirs AlarmNoCountermeasures AlarmSurgical20 AlarmSurgical40 AlarmSurgical80 AlarmFFP220 AlarmFFP240 AlarmFFP280 \
                  -experiment_labels Baseline Surgical-20% Surgical-40% Surgical-80% FFP2-20% FFP2-40% FFP2-80% \
                  -day_x 90 \
                  -baseline_experiment AlarmNoCountermeasures
cd ..
deactivate
