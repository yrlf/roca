

#for method in ccdr cam sam cgnn

for method in rai
# splice letter coil digit1 wdbc usps waveform segment mushroom secstr g241c
#splice secstr letter g241c coil digit1 wdbc usps waveform segment mushroom
# for dataset in wdbc usps
# for dataset in iris krkp splice (0.3 not completed) letter coil(has assertion error) digit1(assertion error) wdbc usps(not completed)

do
    for rate in 0.1 0.2 0.3
    do
        for nt in instance sym pair
        do
            for seed in 1
            do

                for dataset in splice
                # cam cgnn
                do
                  python3 baseline.py --flip_rate_fixed ${rate} --output "rai_splice_baseline.csv" --seed ${seed}  --noise_type ${nt} --dataset ${dataset} --method ${method}
                done

            done
        done
    done
done

