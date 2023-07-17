#!/bin/bash

#default
ngpus="1"
mb_size=8
arch_m3_mlp_lys=8
arch_m3_emb_tbs=100
arch_m3_emb_size=8

while getopts m:l:n:s:g: flag
do
    case "${flag}" in
        m) mb_size=${OPTARG};;
        l) arch_m3_mlp_lys=${OPTARG};;
        n) arch_m3_emb_tbs=${OPTARG};;
        s) arch_m3_emb_size=${OPTARG};;
	g) ngpus=${OPTARG};;
    esac
done

echo "ngpus: $ngpus"
echo "mini batch size: $mb_size"
echo "mlp layers: $arch_m3_mlp_lys"
echo "embedding table num: $arch_m3_emb_tbs"
echo "embedding table siz: $arch_m3_emb_size"

nbatches=4

# Don't change the following setup
dlrm_pt_bin="python3 dlrm_s_pytorch.py"
print_freq=100
emb_size=128
nindices=100
interaction="dot"
#interaction="mean"

_args="--mini-batch-size="${mb_size}\
" --num-batches="${nbatches}\
" --use-m3-bot-mlp --use-m3-top-mlp --arch-m3-mlp-lys="${arch_m3_mlp_lys}\
" --use-m3-emb --debug-m3 --record-gpu-time --arch-m3-emb-tbs="${arch_m3_emb_tbs}\
" --arch-m3-emb-size="${arch_m3_emb_size}\
" --arch-sparse-feature-size="${emb_size}\
" --num-indices-per-lookup="${nindices}\
" --arch-interaction-op="${interaction}\
" --print-freq="${print_freq}\
" --print-time" 
#"--no-mlp-bot"

# GPU Benchmarking
echo "--------------------------------------------"
echo "GPU Benchmarking - running on $ngpus GPUs"
echo "--------------------------------------------"

for _ng in $ngpus
do
    _gpus=$(seq -s, 0 $((_ng-1)))
    cuda_arg="CUDA_VISIBLE_DEVICES=$_gpus"
    echo "-------------------"
    echo "Using GPUS: "$_gpus
    echo "-------------------"
    cmd="$cuda_arg $dlrm_pt_bin $_args --use-gpu"
    echo $cmd
    eval $cmd
done

