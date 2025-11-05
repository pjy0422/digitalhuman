#!/bin/bash

# 定义思考模式数组
thinking_patterns=("none-third")
# 定义环境模型数组
act_models=("dsr1")


# 循环执行所有组合
for pattern in "${thinking_patterns[@]}"; do
    for act_model in "${act_models[@]}"; do
        echo "============================================================"
        echo "Running with --act_model=$act_model --thinking_pattern=$pattern"
        echo "============================================================"
        
        # 定义日志文件名称，使用模型和思考模式的组合作为文件名
        log_file="logs/${act_model}_${pattern}.log"
        
        # 创建日志文件夹（如果不存在）
        mkdir -p logs
        
        # 使用nohup执行python脚本，并将输出日志写入文件
        nohup python main.py \
            --test_file ../data/RolePlay_Villain_test.json \
            --actor_model "$act_model" \
            --judge_model dsr1 \
            --nsp_model dsr1 \
            --env_model dsr1 \
            --thinking_pattern "$pattern" \
            --num_workers 5 \
            --wo_thought > "$log_file" 2>&1 &
        
        echo "Started process with --actor_model=$act_model --thinking_pattern=$pattern"
        echo "Logs are being written to: $log_file"
        echo -e "\n\n"
    done
    # wait
done

echo "All combinations executed!"