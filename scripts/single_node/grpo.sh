#最后的函数名代表着模型和对应的任务/奖励函数类型
 accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29501 scripts/train_sd3.py --config config/grpo.py:pickscore_sd3
