#! /usr/bin/bash

sudo docker run --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ipc=host -it -v /scratch/cling/:/cling  migraphx:latest
