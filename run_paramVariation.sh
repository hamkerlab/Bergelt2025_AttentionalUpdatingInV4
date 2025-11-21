#!/bin/bash

for i in {0..999}
do
    echo "Change number."
    python3 paramtest.py $i

    ./run.sh

    echo "Remove simulation results."
    deldir="Results/paramVariation/$i/trials/"
    rm -rf $deldir

done

