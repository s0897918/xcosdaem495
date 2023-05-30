from transformers import pipeline
import time

model_list = [
    "bigscience/bloom-560m",
    # "bigscience/bloom-1b1"
]

input_sentense = [
    "GPUs play an important role in",
    #  "Currently, AI practitioners have very limited flexibility when choosing a high-performance",
    # "A machine learning system designed for one technology provider’s GPU must be completely reimplemented in order to work on a different provider’s hardware. This lack"
]
device_idx = 1
top_k = 1
token_lengths = [25,50,100,200,300,400,500,600,700,800]
# token_lengths = [20]
do_sample_flag = True
token_length=100
# generator_c = pipeline('text-generation', model=model_list[0], max_new_tokens=token_length+5, min_new_tokens=token_length, do_sample=do_sample_flag, top_k=top_k)
# generator_g = pipeline('text-generation', model=model_list[0], max_new_tokens=token_length+5, min_new_tokens=token_length, do_sample=do_sample_flag, top_k=top_k, device=1)

def time_eclapsed(token_length, sentense, model, device):
    # start_time = time.perf_counter()
    if device == 'CPU':
        generator = pipeline('text-generation', model=model, max_new_tokens=token_length+5, min_new_tokens=token_length, do_sample=do_sample_flag, top_k=top_k)
        start_time = time.perf_counter()
        response = generator(sentense)
        #print(response[0]["generated_text"])
    elif device == 'GPU':
        generator = pipeline('text-generation', model=model, max_new_tokens=token_length+5, min_new_tokens=token_length, do_sample=do_sample_flag, top_k=top_k, device=device_idx)
        start_time = time.perf_counter()
        response = generator(sentense)
        #print(response[0]["generated_text"])
    else:
        print("error: should specify CPU or GPU option before calling")
        return -1

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(str(device) + " Elapsed time: " + "for " + str(model) + " is ",  elapsed_time)

    return elapsed_time

for s in input_sentense:
    for t in token_lengths:
        GPU_time = time_eclapsed(t, s, model_list[0], "GPU")
        CPU_time = time_eclapsed(t, s, model_list[0], "CPU")
        print("speedup for: token = " + str(t) + ", model = "+ model_list[0] + ", speedup = " + str(CPU_time/GPU_time))

