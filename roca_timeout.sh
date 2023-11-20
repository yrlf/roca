#!/bin/bash

#dataset=("xyguassian" "yxguassian" "krkp" "splice" "secstr" "pair0070" "pair0071" "pair0107" "wdbc" "letter" "breastcancer" "coil" "g241c" "iris" "mushroom" "segment" "pair0047" "usps" "waveform" "digit1" "cifar10" "cifar10n_worst" "cifar10n_aggre" "cifar10n_random1" "cifar10n_random2" "cifar10n_random3")

dataset=("cifar10" "xyguassian" "yxguassian" "krkp" "splice" "secstr" "pair0070" "pair0071" "pair0107" "wdbc" "letter" "breastcancer" "coil" "g241c" "iris" "mushroom" "segment" "pair0047" "usps" "waveform" "digit1")
nt=("sym" "instance" "pair")
rate=("0.1" "0.2" "0.3")
#method=("pc" "ges" "ccdr" "fci" "icd" "rai" "lingam" "sam")
method=("1" "2" "3" "4" "5") # 其实是seed, 变量名没有改
timeout_duration=7200 # 设置超时时间，单位为秒

injection="ins"
time_file="time_log.txt"
log_file="roca_timeout_log.txt"
output_file="all_time_compare_270523.csv"

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
            echo "Executing with dataset=$data, noise_type=$noise, rate = $flip_rate, seed = $method" 
            echo "Timeout value is $skip_method"
            if [ "$skip_method" == "true" ]; then
              echo "Skipping dataset=$data"
              echo "Skipping dataset=$data, noise_type=$noise, flip_rate=$flip_rate, seed=$algo" >> "$log_file"
              continue
            fi
            start_time=$(date +%s)
            timeout "${timeout_duration}s" bash -c "python main_method1_v2.py --output '$output_file' --dataset '$data' --noise_type '$noise' --flip_rate_fixed '$flip_rate' --seed '$algo' --noise_injection_type '$injection'"
            end_time=$(date +%s)
            runtime=$((end_time - start_time))
            echo "dataset=$data, noise_type=$noise, filp_rate=$flip_rate, seed=$algo, time=$runtime" >> "$time_file"
            timeout_status=$?
            if [ $timeout_status -eq 124 ]; then
              skip_method="true"
              echo "Execution timed out, skipping to next iteration"
              echo "timeout, dataset=$data, noise_type=$noise, filp_rate=$flip_rate, seed=$algo" >> "$log_file"
              continue
            fi
            if [ $timeout_status -ne 0 ]; then
              #skip_method="true"
              echo "Execution encountered an error, skipping to next iteration"
              echo "error, dataset=$data, noise_type=$noise, filp_rate=$flip_rate, seed=$algo" >> "$log_file"
              continue
            fi
        done
    done
    
  done
done
