import torch
import time
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1-base"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
batch_size = 1
steps = 10
height = 512
width = 512

#pipe = pipe.to("cuda")
start_time = time.perf_counter()
#prompt = "a photo of an astronaut riding a horse on mars"
prompt = "ruins"
images = pipe(prompt, num_inference_steps=steps, num_images_per_prompt=batch_size, height=height, width=width).images
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print("GPU Elapsed time: ", elapsed_time)
