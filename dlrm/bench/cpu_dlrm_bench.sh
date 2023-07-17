#!/bin/bash

#default
mb_size=8
arch_m3_mlp_lys=8
arch_m3_emb_tbs=100
arch_m3_emb_size=8

while getopts m:l:n:s: flag
do
    case "${flag}" in
        m) mb_size=${OPTARG};;
        l) arch_m3_mlp_lys=${OPTARG};;
        n) arch_m3_emb_tbs=${OPTARG};;
        s) arch_m3_emb_size=${OPTARG};;	
    esac
done

echo "mini batch size: $mb_size"
echo "mlp layers: $arch_m3_mlp_lys";
echo "embedding table num: $arch_m3_emb_tbs";
echo "embedding table siz: $arch_m3_emb_size"

nbatches=5

# Don't change the following setup
ncores=1 
nsockets="0"
numa_cmd="numactl --physcpubind=0-$((ncores-1)) -m $nsockets" #run on one socket, without HT
dlrm_pt_bin="python3 dlrm_s_pytorch.py"
print_freq=100
emb_size=128
nindices=100
#interaction="mean"
interaction="dot"

_args="--mini-batch-size="${mb_size}\
" --num-batches="${nbatches}\
" --use-m3-bot-mlp --use-m3-top-mlp --arch-m3-mlp-lys="${arch_m3_mlp_lys}\
" --use-m3-emb --debug-m3 --arch-m3-emb-tbs="${arch_m3_emb_tbs}\
" --arch-m3-emb-size="${arch_m3_emb_size}\
" --arch-sparse-feature-size="${emb_size}\
" --num-indices-per-lookup="${nindices}\
" --arch-interaction-op="${interaction}\
" --print-freq="${print_freq}\
" --print-time"
#--no-mlp-bot"

# CPU Benchmarking
echo "--------------------------------------------"
echo "CPU Benchmarking - running on 1 core"
echo "--------------------------------------------"

cmd="$numa_cmd $dlrm_pt_bin $_args"

echo $cmd
eval $cmd


