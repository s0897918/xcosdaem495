#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#check if extra argument is passed to the test
if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi
#echo $dlrm_extra_option

cpu=0
gpu=1
pt=1
c2=0

ncores=1 #12 #6
nsockets="0"

#ngpus="1 2 4 8"
ngpus="2"

numa_cmd="numactl --physcpubind=0-$((ncores-1)) -m $nsockets" #run on one socket, without HT
dlrm_pt_bin="python dlrm_s_pytorch.py"
dlrm_c2_bin="python dlrm_s_caffe2.py"

data=random #synthetic
print_freq=100
rand_seed=727

c2_net="async_scheduling"

#Model param
#mini_batch_size
mb_size=4 #1024 #512 #256
nbatches=1 #500 #100
bot_mlp="128" #"512-512-64"

arch_m3_bot_mlp=True
arch_m3_top_mlp=True

top_mlp="4096-4096-4096-1" #"1024-1024-1024-1"
emb_size=128
nindices=100

emb="4-4"
#emb="1460-583-10131227-2202608-305-24-12517-633-3-93145-5683-8351593-3194-27-14992-5461306-10-5652-2173-4-7046547-18-15-286181-105-142572"
interaction="dot"
tnworkers=0
tmb_size=16384

_args="--mini-batch-size="${mb_size}\
" --num-batches="${nbatches}\
" --arch-m3-bot-mlp="${arch_m3_bot_mlp}\
" --arch-m3-top-mlp="${arch_m3_top_mlp}\
" --data-generation="${data}\
" --arch-mlp-bot="${bot_mlp}\
" --arch-mlp-top="${top_mlp}\
" --arch-sparse-feature-size="${emb_size}\
" --arch-embedding-size="${emb}\
" --num-indices-per-lookup="${nindices}\
" --arch-interaction-op="${interaction}\
" --numpy-rand-seed="${rand_seed}\
" --print-freq="${print_freq}\
" --print-time"
#\
#" --enable-profiling "

c2_args=" --caffe2-net-type="${c2_net}


# CPU Benchmarking
if [ $cpu = 1 ]; then
  echo "--------------------------------------------"
  echo "CPU Benchmarking - running on $ncores cores"
  echo "--------------------------------------------"
  if [ $pt = 1 ]; then
    outf="model1_CPU_PT_$ncores.log"
    outp="dlrm_s_pytorch.prof"
    echo "-------------------------------"
    echo "Running PT (log file: $outf)"
    echo "-------------------------------"
    #cmd="$numa_cmd $dlrm_pt_bin --mini-batch-size=$mb_size --test-mini-batch-size=$tmb_size --test-num-workers=$tnworkers $_args $dlrm_extra_option > $outf"
    # cmd="$numa_cmd $dlrm_pt_bin --mini-batch-size=$mb_size $_args $dlrm_extra_option"
    cmd="$numa_cmd $dlrm_pt_bin $_args $dlrm_extra_option"
    
    echo $cmd
    eval $cmd
    #min=$(grep "iteration" $outf | awk 'BEGIN{best=999999} {if (best > $7) best=$7} END{print best}')
    #echo "Min time per iteration = $min"
    # move profiling file(s)
    #mv $outp ${outf//".log"/".prof"}
    #mv ${outp//".prof"/".json"} ${outf//".log"/".json"}

  fi
  if [ $c2 = 1 ]; then
    outf="model1_CPU_C2_$ncores.log"
    outp="dlrm_s_caffe2.prof"
    echo "-------------------------------"
    echo "Running C2 (log file: $outf)"
    echo "-------------------------------"
    cmd="$numa_cmd $dlrm_c2_bin --mini-batch-size=$mb_size $_args $c2_args $dlrm_extra_option 1> $outf 2> $outp"
    echo $cmd
    eval $cmd
    min=$(grep "iteration" $outf | awk 'BEGIN{best=999999} {if (best > $7) best=$7} END{print best}')
    #echo "Min time per iteration = $min"
    # move profiling file (collected from stderr above)
    mv $outp ${outf//".log"/".prof"}
  fi
fi

# GPU Benchmarking
if [ $gpu = 1 ]; then
  echo "--------------------------------------------"
  echo "GPU Benchmarking - running on $ngpus GPUs"
  echo "--------------------------------------------"
  for _ng in $ngpus
  do
    # weak scaling
    # _mb_size=$((mb_size*_ng))
    # strong scaling
    _mb_size=$((mb_size*1))
    _gpus=$(seq -s, 0 $((_ng-1)))
    cuda_arg="CUDA_VISIBLE_DEVICES=$_gpus"
    echo "-------------------"
    echo "Using GPUS: "$_gpus
    echo "-------------------"
    if [ $pt = 1 ]; then
      outf="model1_GPU_PT_$_ng.log"
      outp="dlrm_s_pytorch.prof"
      echo "-------------------------------"
      echo "Running PT (log file: $outf)"
      echo "-------------------------------"
      cmd="$cuda_arg $dlrm_pt_bin $_args --use-gpu $dlrm_extra_option"
      # cmd="$cuda_arg $dlrm_pt_bin $_args --use-gpu"
      
      echo $cmd
      eval $cmd
      #min=$(grep "iteration" $outf | awk 'BEGIN{best=999999} {if (best > $7) best=$7} END{print best}')
      echo "Min time per iteration = $min"
      # move profiling file(s)
      #mv $outp ${outf//".log"/".prof"}
      #mv ${outp//".prof"/".json"} ${outf//".log"/".json"}
    fi
    if [ $c2 = 1 ]; then
      outf="model1_GPU_C2_$_ng.log"
      outp="dlrm_s_caffe2.prof"
      echo "-------------------------------"
      echo "Running C2 (log file: $outf)"
      echo "-------------------------------"
      cmd="$cuda_arg $dlrm_c2_bin --mini-batch-size=$_mb_size $_args $c2_args --use-gpu $dlrm_extra_option 1> $outf 2> $outp"
      echo $cmd
      eval $cmd
      min=$(grep "iteration" $outf | awk 'BEGIN{best=999999} {if (best > $7) best=$7} END{print best}')
      echo "Min time per iteration = $min"
      # move profiling file (collected from stderr above)
      mv $outp ${outf//".log"/".prof"}
    fi
  done
fi
