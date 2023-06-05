'''
Helper classes and functions for examples
'''

import os
import io
from pathlib import Path
import json
import deepspeed
import torch
import time
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizerFast

class DSPipeline():
    '''
    Example helper class for comprehending DeepSpeed Meta Tensors, meant to mimic HF pipelines.
    The DSPipeline can run with and without meta tensors.
    '''
    def __init__(self,
                 model_name='bigscience/bloom-3b',
                 dtype=torch.float16,
                 is_meta=True,
                 device=-1,
                 checkpoint_path=None
                 ):
        self.model_name = model_name
        self.dtype = dtype

        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)
        elif device < 0:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device}")

        # the Deepspeed team made these so it's super fast to load (~1 minute), rather than wait 10-20min loading time.
        self.tp_presharded_models = ["microsoft/bloom-deepspeed-inference-int8", "microsoft/bloom-deepspeed-inference-fp16"]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if (is_meta):
            '''When meta tensors enabled, use checkpoints'''
            self.config = AutoConfig.from_pretrained(self.model_name)
            self.repo_root, self.checkpoints_json = self._generate_json(checkpoint_path)

            with deepspeed.OnDevice(dtype=torch.float16, device="meta"):
                self.model = AutoModelForCausalLM.from_config(self.config)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        self.model.eval()

        if self.dtype == torch.float16:
            self.model.half()


    def __call__(self, input_sentences, input_ids, args):

        if (len(input_sentences) >= 1):
            input_ids = self.tokenizer.batch_encode_plus(input_sentences, return_tensors="pt", padding=True)
            input_ids = input_ids["input_ids"]
            input_ids = input_ids.to(self.device)
            self.model.cuda().to(self.device)
            outputs = self.model.generate(input_ids, temperature=0.9, do_sample=False, max_new_tokens=100, pad_token_id=self.tokenizer.eos_token_id)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            print("outputs: ", outputs);
        else:
            global_seed = torch.Generator()
            input_ids = torch.randint(20, 5000, (args.batch_size, 128), generator=global_seed.manual_seed(1000))

            input_ids = input_ids.to(self.device)
            #self.model.cuda().to(self.device)
            q = 128
            a = 32
            batch = args.batch_size
            
            # warm_up
            outputs = self.model.generate(input_ids, temperature=0.9, do_sample=False, min_length=q+1, max_length=q+1, pad_token_id=self.tokenizer.eos_token_id)
            
            start = time.perf_counter()
            outputs = self.model.generate(input_ids, temperature=0.9, do_sample=False, min_length=q+1, max_length=q+1, pad_token_id=self.tokenizer.eos_token_id)
            torch.cuda.synchronize()
            end = time.perf_counter() - start
            query_latency = end
            
            start = time.perf_counter()
            outputs = self.model.generate(input_ids, temperature=0.9, do_sample=False, min_length=q+a+1, max_length=q+a+1, pad_token_id=self.tokenizer.eos_token_id)
            torch.cuda.synchronize()
            end = time.perf_counter() - start
            total_latency = end
            
            answer_lantency = total_latency - query_latency
            token_output_latency = answer_lantency/a * 1000
            tokens_per_second = (1000/token_output_latency)*batch

            if (args.local_rank == 0):
                print("[INFO] model: " + args.name)
                print("batch, query_length, answer_length, query_latency(ms), answer_latency(ms), total_latency(ms), 1-token_output_latency(ms), tokens/second")
                print(str(batch).rjust(len('batch')) + ", " +
                      str(q).rjust(len('query_length')) + ", " +
                      str(a).rjust(len('answer_length')) + ", " +
                      "{:.0f}".format(query_latency * 1000).rjust(len('query_latency(ms)')) + ", " +
                      "{:.0f}".format(answer_lantency * 1000).rjust(len('answer_latency(ms)')) +  ", " +
                      "{:.0f}".format(total_latency * 1000).rjust(len('total_latency(ms)')) + ", " +
                      "{:.0f}".format(token_output_latency).rjust(len('1-token_output_latency(ms)')) + ", " +
                      "{:.0f}".format(tokens_per_second).rjust(len('tokens_second'))) 

            
        return outputs


    def _generate_json(self, checkpoint_path=None):
        if checkpoint_path is None:
            repo_root = snapshot_download(self.model_name,
                                      allow_patterns=["*"],
                                      cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
                                      ignore_patterns=["*.safetensors"],
                                      local_files_only=False,
                                      revision=None)
        else:
            assert os.path.exists(checkpoint_path), f"Checkpoint path {checkpoint_path} does not exist"
            repo_root = checkpoint_path

        if os.path.exists(os.path.join(repo_root, "ds_inference_config.json")):
            checkpoints_json = os.path.join(repo_root, "ds_inference_config.json")
        elif (self.model_name in self.tp_presharded_models):
            # tp presharded repos come with their own checkpoints config file
            checkpoints_json = os.path.join(repo_root, "ds_inference_config.json")
        else:
            checkpoints_json = "checkpoints.json"

            with io.open(checkpoints_json, "w", encoding="utf-8") as f:
                file_list = [str(entry).split('/')[-1] for entry in Path(repo_root).rglob("*.[bp][it][n]") if entry.is_file()]
                data = {"type": "BLOOM", "checkpoints": file_list, "version": 1.0}
                json.dump(data, f)

        return repo_root, checkpoints_json


    def generate_outputs(self,
                         inputs=["test"],
                         bs=1,
                         num_tokens=100,
                         do_sample=False):
        generate_kwargs = dict(min_new_tokens=num_tokens, max_new_tokens=num_tokens, do_sample=do_sample)
        #print("inputs: ", inputs)
        #print ("bs: ", bs)
        
        input_tokens = self.tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True)
        #print("1: input_tokens: ", input_tokens)

        '''
        cling's code to generate a fixed length of query
        '''
        input_ids = torch.randint(20, 5000, (bs, 128))
        attention_mask = torch.randint(2, (bs, 128))
        #print("input_ids: ", input_ids)v
        #print("attention_mask: ", attention_mask)
        input_tokens['input_ids'] = input_ids
        input_tokens['attention_mask'] = attention_mask
        
        #print("2: input_tokens: ", input_tokens)
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(self.device)
                #print("t= ", input_tokens[t])
                #print(self.device)
        self.model.cuda().to(self.device)

        if isinstance(self.tokenizer, LlamaTokenizerFast):
            # NOTE: Check if Llamma can work w/ **input_tokens
            #       'token_type_ids' kwarg not recognized in Llamma generate function
            outputs = self.model.generate(input_tokens.input_ids, **generate_kwargs)
        else:
            outputs = self.model.generate(**input_tokens, **generate_kwargs)
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        #print("outputs len:", len(outputs))
        #print("output: ", outputs);
        return outputs
