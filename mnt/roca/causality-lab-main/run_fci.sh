#!/bin/bash


# for dataset in mushroom splice waveform secstr letter segment g241c xyguassian yxguassian coil digit1 wdbc usps
# for dataset in iris
# for dataset in krkp
for dataset in xyguassian
do
    for rate in 0.01
    do
        for nt in instance
        do
            for seed in 1
            do

                for method in ci
                do
                    python3 run.py --flip_rate_fixed ${rate} --seed ${seed}  --noise_type ${nt} --dataset ${dataset} --method ${method} --output "test.csv"
                done

            done
        done
    done
done
