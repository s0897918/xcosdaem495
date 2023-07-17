#loop=10
mnb_path="check/mnb/"
lys_path="check/lys/"
tbn_path="check/tbn/"
tbs_path="check/tbs/"

bin="gpu_dlrm_bench.sh"

#spetial="-mb-4096"
ngpu=2
flag=$ngpu"-gpu-"

# path=$mnb_path
# for i in {3..12}
# do
#     j=$((2**i))
#     dist=$path$flag$j
#     cmd="bash $bin -m $j -g $ngpu > $dist"
#     echo $cmd
#     eval $cmd
# done


# path=$lys_path
# for i in {8..17}
# do
#     dist=$path$flag$i
#     cmd="bash $bin -l $i -g $ngpu > $dist"
#     echo $cmd
#     eval $cmd
# done

path=$tbn_path
for i in {800..1000..100}
do
    dist=$path$flag$i
    cmd="bash $bin -n $i -g $ngpu > $dist"
    echo $cmd
    eval $cmd
done


# path=$tbs_path
# for i in {128..1280..128}
# do
#     dist=$path$flag$i$spetial
#     cmd="bash $bin -s $i -g $ngpu > $dist"
#     echo $cmd
#     eval $cmd
# done
