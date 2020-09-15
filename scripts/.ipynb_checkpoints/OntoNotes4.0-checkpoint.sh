CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node 3 train.py \
--max_grad_norm 0.5 \
--train_span_method  gold \
--lr 8e-6 \
--max_tokens 450 \
--cls \
--span_layer \
--allow_impossible