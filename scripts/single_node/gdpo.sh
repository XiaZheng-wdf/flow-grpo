CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes=1 --main_process_port 29505 scripts/train_sd3_gdpo.py --config config/pickscore_gdpo.py
