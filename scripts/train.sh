python -u run_train.py --nproc_per_node=4 --master_port=12233 --use_env \
--config_file train_cfg.json \
--notes test-commu
#python -m torch.distributed.launch --nproc_per_node=4 --master_port=12233 --use_env run_train.py \
#--config_file train_cfg.json \
#--notes test-commu
