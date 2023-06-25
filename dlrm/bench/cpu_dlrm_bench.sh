#!/bin/bash


mb_size=2
nbatches=1 


#M3 setup
arch_m3_mlp_lys=3
arch_m3_emb_tbs=2
arch_m3_emb_size=4

# Don't change the following setup
ncores=1 
nsockets="0"
numa_cmd="numactl --physcpubind=0-$((ncores-1)) -m $nsockets" #run on one socket, without HT
dlrm_pt_bin="python dlrm_s_pytorch.py"
print_freq=100
rand_seed=727
emb_size=128
nindices=100
interaction="dot"

_args="--mini-batch-size="${mb_size}\
" --num-batches="${nbatches}\
" --use-m3-bot-mlp --use-m3-top-mlp --arch-m3-mlp-lys="${arch_m3_mlp_lys}\
" --use-m3-emb --debug-m3-no --arch-m3-emb-tbs="${arch_m3_emb_tbs}\
" --arch-m3-emb-size="${arch_m3_emb_size}\
" --arch-sparse-feature-size="${emb_size}\
" --num-indices-per-lookup="${nindices}\
" --arch-interaction-op="${interaction}\
" --numpy-rand-seed="${rand_seed}\
" --print-freq="${print_freq}\
" --print-time"

# CPU Benchmarking
echo "--------------------------------------------"
echo "CPU Benchmarking - running on 1 core"
echo "--------------------------------------------"

cmd="$numa_cmd $dlrm_pt_bin $_args"

echo $cmd
eval $cmd


