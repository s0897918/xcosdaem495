
from transformers import pipeline
import time

model_list = [
    "bigscience/bloom-560m",
    "bigscience/bloom-1b1"
]

def time_eclapsed(model, device):
    start_time = time.perf_counter()
    if device == 'CPU':
        generator = pipeline('text-generation', model=model)
        print(generator("Hello, I'm conscious and"))
        # end_time = time.perf_counter()
        # elapsed_time = end_time - start_time
        # print("CPU Elapsed time: ", elapsed_time)
    elif device == 'GPU':
        generator = pipeline('text-generation', model=model, device=0)
        print(generator("Hello, I'm conscious and"))
        # end_time = time.perf_counter()
        # elapsed_time = end_time - start_time
        # print("GPU Elapsed time: ", elapsed_time)
    else:
        print("error: should specify CPU or GPU option before calling")
        return

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(str(device) + " Elapsed time: " + "for " + str(model) + " is ",  elapsed_time)
    
    return

time_eclapsed(model_list[0], "CPU");
time_eclapsed(model_list[0], "GPU");

time_eclapsed(model_list[1], "CPU");
time_eclapsed(model_list[1], "GPU");


# generator = pipeline('text-generation', model="facebook/opt-350m", do_sample=True)
# generator = pipeline('text-generation', model="facebook/opt-350m")
# print("CPU version for opt-350m:")
# print(generator("Hello, I'm am conscious and"))

# generator = pipeline('text-generation', model="facebook/opt-350m", device=1)
# print("GPU version for opt-350m:")
# print(generator("Hello, I'm am conscious and"))

# generator = pipeline('text-generation', model="facebook/opt-1.3b")
# print("CPU version for opt-1.3b:")
# print(generator("Hello, I'm am conscious and"))

# generator = pipeline('text-generation', model="facebook/opt-1.3b", device=1)
# print("GPU version for opt-1.3b:")
# print(generator("Hello, I'm am conscious and"))
