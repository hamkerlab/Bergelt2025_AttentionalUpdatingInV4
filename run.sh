#!/bin/bash

trial=1
for offset in 200 300 400
do
    echo "Start simulation with turning off top-down attention after $offset ms."
    # 1. generate setup for all trials
    # python3 generateSetup.py
    # 2. simulate one specific trial to record more data (e.g. to plot setup or rates later)
    python3 main_singleTrial.py $trial $offset
    # 3. simulate all (previously) generated trials
    python3 main.py -1 $offset
    echo "Finished simulation."

    echo "Start evaluation."
    cd evaluation
    # 4. generate all Figures (with needed data, which are stored for further usage)
    python3 evaluate.py $trial $offset
    echo "Finished evaluation."
    cd ..
done