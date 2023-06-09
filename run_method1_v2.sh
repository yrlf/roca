#!/bin/bash

#for dataset in xyguassian yxguassian balancescale krkp waveform splice cifar10 cifar10n_worst cifar10n_aggre cifar10n_random1 cifar10n_random2 cifar10n_random3

#for dataset in krkp waveform splice iris secstr wdbc letter mushroom segment coil digit1 usps g241c xyguassian yxguassian 
#for dataset in xyguassian krkp waveform splice iris secstr wdbc letter mushroom segment coil digit1 g241c
#for dataset in cifar10n_random1 cifar10n_random2 cifar10n_random3 cifar10n_worst cifar10n_aggre
for dataset in xyguassian yxguassian
do
    for nt in instance sym pair
    do
        for seed in 1 2 3 4 5 
        do
            for rate in 0.3
            do
                python3 main_method1_v2.py --output "xy_yx_0d5_plotting" --seed ${seed}  --noise_type ${nt} --dataset ${dataset} --flip_rate_fixed ${rate} --noise_injection_type "ins"
                 
            done             
        done
    done
done
