#!/bin/bash

trial=1
AP="[ 0. -7.]"
echo "Start simulation with fixed attention position at $AP."
# 1. generate sup-setup for all trials with given AP
python3 extractSubSetup.py "$AP"
# 2. simulate one specific trial to record more data (e.g. to plot setup or rates later)
python3 main_singleTrial.py $trial "$AP"
# 3. simulate all (previously) generated trials
python3 main.py -1 "$AP"
echo "Finished simulation."

echo "Start evaluation."
cd evaluation
# 4. generate all Figures (with needed data, which are stored for further usage)
python3 evaluate.py $trial "$AP"
echo "Finished evaluation."