#!/bin/bash


for dataset in mushroom splice waveform secstr letter segment g241c xyguassian yxguassian coil digit1 wdbc usps
# for dataset in iris
# for dataset in krkp

do
    for rate in 0.01 0.1 0.2 0.3
    do
        for nt in instance pair sym
        do
            for seed in 1
            do

                for method in fci icd rai ci
                do
                    python3 run.py --flip_rate_fixed ${rate} --seed ${seed}  --noise_type ${nt} --dataset ${dataset} --method ${method} --output "all_dataset_baseline.csv"
                done

            done
        done
    done
done
