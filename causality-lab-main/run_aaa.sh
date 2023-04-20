#!/bin/bash

for dataset in iris krkp mushroom splice waveform secstr letter segment g241c 
# for dataset in wdbc usps
# for dataset in coil digit1 

do
    for rate in 0.1 0.3
    do
        for nt in instance pair sym
        do
            for seed in 1
            do

                for method in pc ges gies lingam
                do
                    python3 aaa.py --flip_rate_fixed ${rate} --seed ${seed}  --noise_type ${nt} --dataset ${dataset} --method ${method}
                done

            done
        done
    done
done
