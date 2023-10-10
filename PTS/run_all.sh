#! /bin/bash

set -eu

declare -A PCS=( [1]="Periodical" [2]="Disposable" [3]="Distance" [4]="Random" [5]="Car2Car")

for fq in {1..1}
do
    for pc in {1..5}
    do
        echo -e "Executing the PTF with frequency $fq and policy $pc -> \"${PCS[$pc]}\"" 
        python3 tracker.py -dir results/ -fq $fq -pc $pc
    done
done
