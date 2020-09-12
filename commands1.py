import os
import time 

cmd = "CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py  --train_path /home/wangnan/mrc4ner/datasets/ChineseMSRA/mrc-ner.train  --dataset_tag ChineseMSRA"
cmd = "CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 train.py  --debug  --eval  --loss_sampler_epoch 1000  --reload --epochs 100"
cmd = "CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node 2 train.py  --debug  --eval  --loss_sampler_epoch 3  --reload --epochs 100"

os.system(cmd)
