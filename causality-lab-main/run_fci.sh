#!/bin/bash


#for dataset in  splice secstr letter g241c coil digit1 wdbc usps xyguassian yxguassian waveform segment
for dataset in cifar10
# for dataset in iris krkp
# for dataset in mushroom
#for dataset in xyguassian
do
    for rate in 0.01 0.1 0.2 0.3
    do
        for nt in sym instance pair
        do
            for seed in 1
            do

                for method in fci
                # icd rai ci
                do
                    python3 run.py --flip_rate_fixed ${rate} --seed ${seed}  --noise_type ${nt} --dataset ${dataset} --method ${method} --output "baseline_fci.csv"
                done

            done
        done
    done
done
