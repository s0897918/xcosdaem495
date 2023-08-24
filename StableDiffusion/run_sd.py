from argparse import ArgumentParser
import time
import sys
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

parser = ArgumentParser()
parser.add_argument("--warm_up", default=1, type=int, help="warm up")
parser.add_argument("--generation", default=1, type=int, help="generation")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--img_height", default=512, type=int, help="batch size")
parser.add_argument("--img_width", default=512, type=int, help="batch size")
parser.add_argument("--inference_step", default=100, type=int, help="number of inference steps")
parser.add_argument("--model_id", default="stabilityai/stable-diffusion-2-1-base", type=str, help="model card id")
parser.add_argument("--data_type", default=torch.float16, type=str, help="data-type")
parser.add_argument("--torch_compile_mode", default="default", type=str, choices=["default", "reduce-overhead", "max-autotune"], help="pytorch compile mode")
parser.add_argument("--prompt", default="a photo of an astronaut riding a horse on mars", type=str, help="prompt text")

args = parser.parse_args()

def test_sd2(args):
    model_id = args.model_id
    batch_size = args.batch_size
    height = args.img_height
    width = args.img_width
    inference_step = args.inference_step
    data_type = args.data_type
    prompt = args.prompt
    warm_up = args.warm_up
    generation = args.generation
    compile_mode = args.torch_compile_mode
    

    print(args)

    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    retry = 5
    while retry >= 5:
        retry -= 1
        try:
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            break
        except Exception:
            print("Exception")
            pass
        print('pipe failed, retry...')
        time.sleep(1)
    if '2-1' in model_id:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to("cuda")

    pipe.unet = torch.compile(pipe.unet, mode=compile_mode)

    st = torch.cuda.Event(enable_timing=True)
    ed = torch.cuda.Event(enable_timing=True)

    # warm up
    print("compile the model:")
    for _ in range(warm_up):
        image = pipe(prompt, num_inference_steps=inference_step, num_images_per_prompt=batch_size, height=height, width=width).images[0]

    torch.cuda.nvtx.range_push(model_id)
    
    st.record()
    print("run for image generation:")
    for _ in range(generation):
        image = pipe(prompt, num_inference_steps=inference_step, num_images_per_prompt=batch_size, height=height, width=width).images[0]
    ed.record()
    ed.synchronize()
    
    ms = st.elapsed_time(ed) / generation
    torch.cuda.nvtx.range_pop()
    print('Model: ' + model_id + '\nGeneration time: {:2f}'.format(ms) + '(ms)' +' \nPerformance: {:.2f}'.format(inference_step / ms * 1000) + '(iter/s)')
    # image.save("astronaut_rides_horse.png")

if __name__ == '__main__':
    test_sd2(args)
    #test_sd2('stabilityai/stable-diffusion-2-1-base')
    #test_sd2('stabilityai/stable-diffusion-2-1')
    #test_sd2('runwayml/stable-diffusion-v1-5')

