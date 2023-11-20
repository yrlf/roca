#!/bin/bash

#for dataset in xyguassian yxguassian balancescale krkp waveform splice cifar10 cifar10n_worst cifar10n_aggre cifar10n_random1 cifar10n_random2 cifar10n_random3

for dataset in xyguassian
# krkp waveform splice 
# secstr wdbc letter mushroom segment coil digit1 usps g241c 
#for dataset in xyguassian krkp waveform splice iris secstr wdbc letter mushroom segment coil digit1 g241c
#for dataset in cifar10n_random1 cifar10n_random2 cifar10n_random3 cifar10n_worst cifar10n_aggre
# for dataset in xyguassian yxguassian krkp splice secstr pair0070 pair0071 pair0107 wdbc letter breastcancer coil g241c iris mushroom segment segment usps pair0047 digit1 cifar10 cifar10n_random1 cifar10n_random2 cifar10n_random3 cifar10n_worst cifar10n_aggre
#for dataset in krkp secstr pair0070 pair0071 pair0107 xyguassian 
# causal: secstr krkp splice
do
    for nt in pair
    do
        for seed in 1 2 3
        do
            for rate in 0.1 0.2 0.3
            do
                python3 method1.py --output "new_xy_pair" --seed ${seed}  --noise_type ${nt} --dataset ${dataset} --flip_rate_fixed ${rate} --noise_injection_type "ins"
                 
            done             
        done
    done
done
