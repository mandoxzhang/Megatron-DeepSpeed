#! /bin/bash

# Runs the "345M" parameter model


# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=8
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


DATA_PATH='./'
CHECKPOINT_PATH=./

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
export NCCL_PROTOS=2
export NCCL_IB_HCA=^mlx5_3:1

# python -m torch.distributed.launch $DISTRIBUTED_ARGS \
deepspeed --hostfile /home/mccxadmin/yehua/ds/Megatron-DeepSpeed/examples/MoE/hostfile \
       pretrain_gpt.py \
       --deepspeed \
       --deepspeed_config /home/mccxadmin/yehua/ds/Megatron-DeepSpeed/mccl_example/ds_zero_offload.config \
       --no-pipeline-parallel \
       --num-layers 4 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 2 \
       --global-batch-size 8 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 500000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file gpt2-vocab.json \
       --merge-file gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend mccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --checkpoint-activations \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --tensor-model-parallel-size 8 \
       --zero-stage=3
