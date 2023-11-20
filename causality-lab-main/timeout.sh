#!/bin/bash

dataset=("xyguassian" "yxguassian" "krkp" "splice" "secstr" "pair0070" "pair0071" "pair0107" "wdbc" "letter" "breastcancer" "segment" "pair0047" "iris"  "mushroom"  "waveform" )
#dataset=("coil" "g241c""digit1" "usps" )
# segment
nt=("pair")
rate=("0.1")
method=("icd" "lingam" "sam" "rai")
#method=("fci" "pc" "ges" "gies" "ccdr" "icd" "rai" "lingam" "sam")
timeout_duration=3600 # 设置超时时间，单位为秒

log_file="timeout_log.txt"
time_file="time_log_baseline.txt"
output_file="baseline_results/all_baseline.csv"

# 循环执行参数组合

for algo in "${method[@]}"
do
  for data in "${dataset[@]}"
  do
    skip_method="false"
    echo "reset skip_dataset to false"
        
    for noise in "${nt[@]}"
    do
        if [ "$noise" == "instance" ]; then
            rate_extra=("0.0" "${rate[@]}")
        else
            rate_extra=("${rate[@]}")
        fi

        for flip_rate in "${rate_extra[@]}"
        do
            echo "Executing with dataset=$data, noise_type=$noise, rate = $flip_rate, algo = $method" 
            echo "TimeOUT value is $skip_method"
            if [ "$skip_method" == "true" ]; then
              echo "Skipping dataset=$data"
              echo "Skipping dataset=$data, noise_type=$noise, flip_rate=$flip_rate, method=$algo" >> "$log_file"
              continue
            fi
            start_time=$(date +%s)
            timeout "${timeout_duration}s" bash -c "python baseline.py --output '$output_file' --dataset '$data' --noise_type '$noise' --flip_rate_fixed '$flip_rate' --method '$algo'"
            timeout_status=$?
            if [ $timeout_status -eq 124 ]; then
              skip_method="true"
              echo "Execution timed out, skipping to next iteration"
              echo "timeout, dataset=$data, noise_type=$noise, filp_rate=$flip_rate, method=$algo" >> "$log_file"
              continue
            fi
            if [ $timeout_status -ne 0 ]; then
              #skip_method="true"
              echo "Execution encountered an error, skipping to next iteration"
              echo "error, dataset=$data, noise_type=$noise, filp_rate=$flip_rate, method=$algo" >> "$log_file"
              continue
            fi
            end_time=$(date +%s)
            runtime=$((end_time - start_time))
            echo "dataset=$data, noise_type=$noise, filp_rate=$flip_rate, seed=$algo, time=$runtime" >> "$time_file"
        done
    done
    
  done
done
