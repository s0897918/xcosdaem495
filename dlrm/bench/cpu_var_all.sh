#loop=10
mnb_path="check/mnb/"
lys_path="check/lys/"
tbn_path="check/tbn/"
tbs_path="check/tbs/"

extra="-mb-4096"
flag="1-core-"

# path=$mnb_path
# for i in {3..12}
# do
#     j=$((2**i))
#     dist=$path$j
#     cmd="bash cpu_dlrm_bench.sh -m $j > $dist"
#     echo $cmd
#     eval $cmd
# done


# path=$lys_path
# for i in {8..17}
# do
#     dist=$path$i
#     cmd="bash cpu_dlrm_bench.sh -l $i > $dist"
#     echo $cmd
#     eval $cmd
# done

# path=$tbn_path
# for i in {100..1000..100}
# do
#     dist=$path$i
#     cmd="bash cpu_dlrm_bench.sh -n $i > $dist"
#     echo $cmd
#     eval $cmd
# done


path=$tbs_path
for i in {128..1280..128}
do
    dist=$path$flag$i$extra
    cmd="bash cpu_dlrm_bench.sh -s $i > $dist"
    echo $cmd
    eval $cmd
done
