#!/bin/bash

trial=1

echo "Start simulation."
# 1. generate setup for all trials
python3 generateSetup.py
# 2. simulate one specific trial to record more data (e.g. to plot setup or rates later)
python3 main_singleTrial.py $trial
# 3. simulate all (previously) generated trials
python3 main.py
echo "Finished simulation."

echo "Start evaluation."
cd evaluation
# 4. generate all Figures (with needed data, which are stored for further usage)
python3 evaluate.py $trial
echo "Finished evaluation."
