#! /usr/bin/bash

sudo docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ipc=host -it -v /scratch/cling/:/cling  rocm/pytorch:rocm5.4.3_ubuntu20.04_py3.8_pytorch_1.10.1 
