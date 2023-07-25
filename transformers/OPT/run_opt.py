import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import sys

input_sentence_template = ["Hello, I'm am conscious and"]
batch_exp = [1, 4]
#batch_exp = [1]
device = torch.device("cuda:0")
dtype = torch.float32
#device = torch.device("cpu")
model_list = ["facebook/opt-1.3b", "bigscience/bloom-1b1"]
#model_list = ["facebook/opt-350m", "facebook/opt-1.3b", "bigscience/bloom-560m", "bigscience/bloom-1b1"]

for i in range(0, len(model_list)):
    model_name = model_list[i]
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.eval()


    print("[INFO] model: " + model_name + " on GPU")
    print("batch, query_length, answer_length, query_to_ids_latency(ms), gen_answer_ids_latency(ms), ids_to_answer_latency(ms), total_latency(ms), 1-token_latency(ms), tokens/second")

    for b in batch_exp:
        batch = b
        q = 8
        a = 2048
        input_sentences = batch * input_sentence_template
        start = time.perf_counter()
        input_ids = tokenizer(input_sentences, return_tensors="pt").input_ids
        query_to_ids_latency = time.perf_counter() - start

        start = time.perf_counter()
        input_ids = input_ids.to(device)
        model.to(device)
        output_ids = model.generate(input_ids, do_sample=True, min_length=a, max_length=a)
        torch.cuda.synchronize()
        gen_answer_ids_latency = time.perf_counter() - start

        start = time.perf_counter()
        outputs = tokenizer.batch_decode(output_ids)
        #print(outputs)
        ids_to_answer_latency = time.perf_counter() - start

        total_latency = query_to_ids_latency + gen_answer_ids_latency + ids_to_answer_latency
        token_output_latency = total_latency/a * 1000
        tokens_per_second = (1000/token_output_latency)*batch

        print(str(batch).rjust(len('batch')) + ", " +
              str(q).rjust(len('query_length')) + ", " +
              str(a).rjust(len('answer_length')) + ", " +
              "{:.0f}".format(query_to_ids_latency * 1000).rjust(len('query_to_ids_latency(ms)')) + ", " +
              "{:.0f}".format(gen_answer_ids_latency * 1000).rjust(len('gen_answer_ids_latency(ms)')) +  ", " +
              "{:.0f}".format(ids_to_answer_latency * 1000).rjust(len('ids_to_answer_latency(ms)')) +  ", " +
              "{:.0f}".format(total_latency * 1000).rjust(len('total_latency(ms)')) + ", " +
              "{:.0f}".format(token_output_latency).rjust(len('1-token_latency(ms)')) + ", " +
              "{:.0f}".format(tokens_per_second).rjust(len('tokens_second')))
