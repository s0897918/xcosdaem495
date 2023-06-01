import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_lists = [
"facebook/opt-350m",
#"facebook/opt-1.3b",
#"facebook/opt-6.7b",
#"facebook/opt-13b",
#"facebook/opt-66b",
]


distributed = False
if (len(sys.argv) != 1):
  if(sys.argv[1] == 'gpus'):
    distributed = True
  else:
    print("Specify with para 'gpus' if multiple GPUs are used, sys.exit()")
    sys.exit()


for model_list in model_lists:
  model_dev = model_list.split("/")[0]
  model_name = model_list.split("/")[1]
  device = "cuda:2"
  query_answer_length = {128:32}
  d_type = torch.float16

  if  distributed == True:
      model_name += " in distributed version"
      model = AutoModelForCausalLM.from_pretrained(model_list, torch_dtype=d_type, device_map = 'auto')
  else:
      model = AutoModelForCausalLM.from_pretrained(model_list, torch_dtype=d_type).cuda(device)

  tokenizer = AutoTokenizer.from_pretrained(model_list, use_fast=False)

  warm_up = True
    
  print("[INFO] model: " + model_name)
  print("batch, query_length, answer_length, query_latency(ms), answer_latency(ms), total_latency(ms), 1-token_output_latency(ms), tokens/second")

  model.eval()
  batch_exp = 7
  
  for q, a in query_answer_length.items():
    for b in range(0, batch_exp):
      batch = 2**b

      if distributed == True:
        input_ids = torch.randint(20, 50000, (batch, q)).cuda()
      else:
        input_ids = torch.randint(20, 50000, (batch, q)).cuda(device)
      
      if warm_up == True:
        gen_tokens = model.generate(input_ids, temperature=0.9, do_sample=False, min_length = q + 1, max_length = q + 1, pad_token_id=tokenizer.eos_token_id)
        warm_up = False
      
      start = time.perf_counter()
      gen_tokens = model.generate(input_ids, temperature=0.9, do_sample=False, min_length = q + 1, max_length = q + 1, pad_token_id=tokenizer.eos_token_id)
      end = time.perf_counter() - start
      query_latency = end

      start = time.perf_counter()
      gen_tokens = model.generate(input_ids, temperature=0.9, do_sample=False, min_length = q + a + 1, max_length = q + a + 1, pad_token_id=tokenizer.eos_token_id)
      end = time.perf_counter() - start
      total_latency = end

      answer_lantency = total_latency - query_latency
      token_output_latency = answer_lantency/a * 1000
      tokens_per_second = (1000/token_output_latency)*batch
      
      print(str(batch).rjust(len('batch')) + ", " +
            str(q).rjust(len('query_length')) + ", " +
            str(a).rjust(len('answer_length')) + ", " +
            "{:.0f}".format(query_latency * 1000).rjust(len('query_latency(ms)')) + ", " +
            "{:.0f}".format(answer_lantency * 1000).rjust(len('answer_latency(ms)')) +  ", " +
            "{:.0f}".format(total_latency * 1000).rjust(len('total_latency(ms)')) + ", " +
            "{:.0f}".format(token_output_latency).rjust(len('1-token_output_latency(ms)')) + ", " +
            "{:.0f}".format(tokens_per_second).rjust(len('tokens_second'))) 
