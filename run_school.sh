#!/bin/bash

./abm_ensemble.sh -expdir AlarmNoCountermeasures
./abm_ensemble.sh -expdir AlarmSurgical20
./abm_ensemble.sh -expdir AlarmSurgical40
./abm_ensemble.sh -expdir AlarmSurgical80
./abm_ensemble.sh -expdir AlarmFFP220
./abm_ensemble.sh -expdir AlarmFFP240
./abm_ensemble.sh -expdir AlarmFFP280

cd resources
python postprocessing.py -experiment_dirs $EXPERIMENT_DIR
python barplot.py -experiment_dirs AlarmSurgical20 AlarmSurgical40 AlarmSurgical80 AlarmFFP220 AlarmFFP240 AlarmFFP280 \
                  -experiment_labels Surgical-20% Surgical-40% Surgical-80% FFP2-20% FFP2-40% FFP2-80% \
                  -day_x 30 \
                  -baseline_experiment AlarmNoCountermeasures
cd ..