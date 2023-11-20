#!/bin/bash

for dataset in splice
# for dataset in wdbc usps
# for dataset in coil digit1 

do
    for rate in 0.2 0.4
    do
        for nt in instance pair sym
        do
            for seed in 1
            do

                for method in gies lingam pc  
                do
                    python3 aaa.py --flip_rate_fixed ${rate} --seed ${seed}  --noise_type ${nt} --dataset ${dataset} --method ${method}
                done

            done
        done
    done
done
