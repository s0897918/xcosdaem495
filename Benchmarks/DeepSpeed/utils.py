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
        print("device: ", device)
        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)
        elif device < 0:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device}")

        # the Deepspeed team made these so it's super fast to load (~1 minute), rather than wait 10-20min loading time.
        #self.tp_presharded_models = ["microsoft/bloom-deepspeed-inference-int8", "microsoft/bloom-deepspeed-inference-fp16"]

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


    def __call__(self, input_sentence_template, args):

        if (len(input_sentence_template) >= 1):

            batch_exp = [1, 2, 4, 8, 16, 32, 64]

            print("[INFO] model: " + args.name)
            print("batch, query_length, answer_length, query_to_ids_latency(ms), gen_answer_ids_latency(ms), ids_to_answer_latency(ms), total_latency(ms), 1-token_latency(ms), tokens/second")
            
            for b in batch_exp:
                
                batch = b
                q = 8
                a = 2048

                input_sentences = batch * input_sentence_template

                start = time.perf_counter()
                input_ids = self.tokenizer(input_sentences, return_tensors="pt").input_ids
                query_to_ids_latency = time.perf_counter() - start
                
                # print(input_ids)
                start = time.perf_counter()
                input_ids = input_ids.to(self.device)
                self.model.cuda().to(self.device)
                output_ids = self.model.generate(input_ids, do_sample=True, min_length=a, max_length=a)
                torch.cuda.synchronize()
                gen_answer_ids_latency = time.perf_counter() - start

                
                # print(output_ids)
                start = time.perf_counter()
                outputs = self.tokenizer.batch_decode(output_ids)
                ids_to_answer_latency = time.perf_counter() - start

                total_latency = query_to_ids_latency + gen_answer_ids_latency + ids_to_answer_latency
                token_output_latency = total_latency/a * 1000
                
                tokens_per_second = (1000/token_output_latency)*batch

                if (args.local_rank == 0):
                    print(str(batch).rjust(len('batch')) + ", " +
                          str(q).rjust(len('query_length')) + ", " +
                          str(a).rjust(len('answer_length')) + ", " +
                          "{:.0f}".format(query_to_ids_latency * 1000).rjust(len('query_to_ids_latency(ms)')) + ", " +
                          "{:.0f}".format(gen_answer_ids_latency * 1000).rjust(len('gen_answer_ids_latency(ms)')) +  ", " +
                          "{:.0f}".format(ids_to_answer_latency * 1000).rjust(len('ids_to_answer_latency(ms)')) +  ", " +
                          "{:.0f}".format(total_latency * 1000).rjust(len('total_latency(ms)')) + ", " +
                          "{:.0f}".format(token_output_latency).rjust(len('1-token_latency(ms)')) + ", " +
                          "{:.0f}".format(tokens_per_second).rjust(len('tokens_second'))) 

        else:
            q = 128
            a = 32
            batch_exp = 10
            print("[INFO] model: " + args.name)
            print("batch, query_length, answer_length, query_latency(ms), answer_latency(ms), total_latency(ms), 1-token_output_latency(ms), tokens/second")
            self.model.cuda().to(self.device)

            for b in range (0, batch_exp):
                batch = 2**b
                global_seed = torch.Generator()
                input_ids = torch.randint(20, 5000, (batch, 128), generator=global_seed.manual_seed(1000))
                input_ids = input_ids.to(self.device)
                
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



