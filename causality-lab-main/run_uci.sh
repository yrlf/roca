#!/bin/bash


for rate in 0.2 0.4 0.3  
do
    for nt in instance pair sym
    do
        for seed in 1
        do

            for dataset in splice
            do
                python3 run.py --flip_rate_fixed ${rate} --seed ${seed}  --noise_type ${nt} --dataset ${dataset} > ${dataset}_${nt}_flip_rate${rate}_seed${seed}.out
            done

        done
    done
done

