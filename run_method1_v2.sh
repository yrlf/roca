#!/bin/bash

#for dataset in xyguassian yxguassian balancescale krkp waveform splice cifar10 cifar10n_worst cifar10n_aggre cifar10n_random1 cifar10n_random2 cifar10n_random3

for dataset in krkp waveform splice iris secstr wdbc letter mushroom segment coil digit1 usps g241c xyguassian yxguassian

#for dataset in xyguassian krkp waveform splice iris secstr wdbc letter mushroom segment coil digit1 g241c
#for dataset in cifar10n_random1 cifar10n_random2 cifar10n_random3 cifar10n_worst cifar10n_aggre
# for dataset in xyguassian yxguassian krkp splice secstr pair0070 pair0071 pair0107 wdbc letter breastcancer coil g241c iris mushroom segment segment usps pair0047 digit1 cifar10 cifar10n_random1 cifar10n_random2 cifar10n_random3 cifar10n_worst cifar10n_aggre
#for dataset in xyguassian 
do
    for nt in instance # sym pair
    do
        for seed in $(seq 1 30) #30)
        do
            for rate in 0.01 0.1 0.2 0.3
            do
                python3 main_method1_v2.py --output "test" --seed ${seed}  --noise_type ${nt} --dataset ${dataset} --flip_rate_fixed ${rate} --noise_injection_type "sym"
                 
            done             
        done
    done
done
