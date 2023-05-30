from transformers import pipeline
import time

model_list = [
    "facebook/opt-350m",
    "bigscience/bloom-560m"
]

input_sentense = [
    "GPUs play an important role in",
]
device_idx = 0
top_k = 1
token_lengths = [25]
do_sample_flag = True
def time_eclapsed(token_length, sentense, model, device):
    if device == 'CPU':
        generator = pipeline('text-generation', model=model, max_new_tokens=token_length, min_new_tokens=token_length, do_sample=do_sample_flag, top_k=top_k)
        start_time = time.perf_counter()
        response = generator(sentense)
        print (response)
    elif device == 'GPU':
        generator = pipeline('text-generation', model=model, max_new_tokens=token_length, min_new_tokens=token_length, do_sample=do_sample_flag, top_k=top_k, device=device_idx)
        start_time = time.perf_counter()
        response = generator(sentense)
        print (response)
    else:
        print("error: should specify CPU or GPU option before calling")
        return -1

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(str(device) + " Elapsed time: " + "for " + str(model) + " is ",  elapsed_time)


    return elapsed_time
for m in model_list:
    for s in input_sentense:
        for t in token_lengths:
            GPU_time = time_eclapsed(t, s, m, "GPU")
            CPU_time = time_eclapsed(t, s, m, "CPU")
            print("speedup for: token = " + str(t) + ", model = "+ m + ", speedup = " + str(CPU_time/GPU_time))
