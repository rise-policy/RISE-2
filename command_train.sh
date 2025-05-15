# teleop
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_addr 127.0.0.1 --master_port 23333 --nproc_per_node 4 --nnodes 1 --node_rank 0 train.py --data_path data/collect_toys --ckpt_dir logs/collect_toys --config configs/dual_teleop_dino.yaml

# wild
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_addr 127.0.0.1 --master_port 23333 --nproc_per_node 4 --nnodes 1 --node_rank 0 train.py --data_path data/collect_toys_wild --ckpt_dir logs/collect_toys_wild --config configs/dual_wild_dino.yaml
