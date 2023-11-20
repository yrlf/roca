

#for method in ccdr cam sam cgnn

timeout_duration = 10

for dataset in coil g241c usps
#coil g241c usps
# splice letter coil digit1 wdbc usps waveform segment mushroom secstr g241c
#splice secstr letter g241c coil digit1 wdbc usps waveform segment mushroom
# for dataset in wdbc usps
# for dataset in iris krkp splice (0.3 not completed) letter coil(has assertion error) digit1(assertion error) wdbc usps(not completed)

do
    for nt in instance sym pair
    do
        for seed in 1
        do

            for method in icd
            # cam cgnn
            do
                for rate in 0.1 0.2 0.3
                do
                  echo "Executing with dataset=$dataset, method=$method, noise=$nt, rate=$rate"
                  timeout ${timeout_duration} python baseline.py --flip_rate_fixed ${rate} --output "baseline_results/all_baseline.csv" --seed ${seed}  --noise_type ${nt} --dataset ${dataset} --method ${method} 
                  # timeout_status=$?
                  # if [ $timeout_status -eq 124 ]; then
                  #     echo "Execution timed out, skipping to next iteration"
                  #     continue
                  # fi
                  # if [ $timeout_status -ne 0 ]; then
                  #     echo "Execution encountered an error, skipping to next iteration"
                  #     continue
                  # fi
                done
            done

        done
    done
    
done
