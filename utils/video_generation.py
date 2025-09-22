import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXImageToVideoPipeline, CogVideoXTransformer3DModel
from diffusers import WanImageToVideoPipeline, AutoencoderKLWan, ModularPipeline
from diffusers import I2VGenXLPipeline
from diffusers import DiffusionPipeline
from diffusers import  HunyuanVideoTransformer3DModel, HunyuanVideoImageToVideoPipeline
from diffusers.utils import export_to_video, load_image, load_video
from transformers import T5EncoderModel, CLIPVisionModel
import os
import gc
import json
import numpy as np

def choose_resolution(choice: str) -> tuple:
    resolutions = {
        "240p": (240,432),
        "360p": (360,640),
        "480p": (480,720),
        "720p": (720,1280),
    }
    return resolutions.get(choice)

@torch.no_grad
def test_CogVideoX(input_image_path, out_video_path, resolution=(720,480), **args):
    model_id = "THUDM/CogVideoX-5b-I2V"

    transformer = CogVideoXTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.float16)
    text_encoder = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float16)
    vae = AutoencoderKLCogVideoX.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        model_id,
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae,
        torch_dtype=torch.float16,
    )
    pipe.to(args['device'])
    pipe.enable_sequential_cpu_offload()
    save_video_path = os.path.join(out_video_path,'CogVideoX')
    os.makedirs(save_video_path, exist_ok=True)
    filenames = os.listdir(input_image_path)
    negative_prompt = "Vibrant colors, overexposed, static, blurry details, subtitles, style, artwork, painting, scene, static, overall grayish, poor quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, deformed limbs, fused fingers, static scene, cluttered background, three legs, many people in the background, walking backward"
    with open(args['prompt'], 'r') as f:
        prompts = json.load(f)
    for filename in filenames[:args['test_num']]:
        image = load_image(os.path.join(input_image_path, filename))
        image = image.resize(resolution)
        prompt = prompts[filename.split('.')[0]]
        video = pipe(image=image, prompt=prompt, guidance_scale=args['guidance_scale'], 
                     use_dynamic_cfg=True, num_inference_steps=args['steps'],
                     negative_prompt=negative_prompt).frames[0]

        export_to_video(video, os.path.join(save_video_path,f"{filename.split('.')[0]}.mp4")
                        , fps=args['fps'], quality=args['quality'])
    gc.collect()
    torch.cuda.empty_cache()

@torch.no_grad
def test_Wan(input_image_path,out_video_path,resolution=(720,480),**args):
    # Wan2.1
    model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float16)
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
    pipe = WanImageToVideoPipeline.from_pretrained(model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.float16)
    pipe.to(args['device'])
    pipe.enable_model_cpu_offload()
    save_video_path = os.path.join(out_video_path,'Wan')
    os.makedirs(save_video_path, exist_ok=True)
    filenames = os.listdir(input_image_path)
    generator = torch.Generator(device=args['device']).manual_seed(0)
    negative_prompt = "Vibrant colors, overexposed, static, blurry details, subtitles, style, artwork, painting, scene, static, overall grayish, poor quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, deformed limbs, fused fingers, static scene, cluttered background, three legs, many people in the background, walking backward"
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    max_area = resolution[0] * resolution[1]
    for filename in filenames[:args['test_num']]:
        image = load_image(os.path.join(input_image_path, filename))
        aspect_ratio = image.height / image.width
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize(resolution)
        prompt = args['prompt']
        video = pipe(image=image, prompt=prompt, guidance_scale=args['guidance_scale'], num_frames=49,
                      num_inference_steps=args['steps'], height=height, width=width, generator=generator,
                      negative_prompt=negative_prompt ).frames[0]

        export_to_video(video, os.path.join(save_video_path,f"{filename.split('.')[0]}.mp4")
                        , fps=args['fps'], quality=args['quality'])
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad
def test_Hunyuan(input_image_path,out_video_path,resolution=(720,480),**args):
    # https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V
    # quantize weights to int4 with bitsandbytes
    save_video_path = os.path.join(out_video_path,'Hunyuan')
    os.makedirs(save_video_path, exist_ok=True)
    filenames = os.listdir(input_image_path)
    model_id = "hunyuanvideo-community/HunyuanVideo-I2V"
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    pipeline = HunyuanVideoImageToVideoPipeline.from_pretrained(
        model_id, transformer=transformer, torch_dtype=torch.float16
    )

    # model-offloading and tiling
    pipeline.enable_model_cpu_offload()
    pipeline.vae.enable_tiling()
    negative_prompt = "Vibrant colors, overexposed, static, blurry details, subtitles, style, artwork, painting, scene, static, overall grayish, poor quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, deformed limbs, fused fingers, static scene, cluttered background, three legs, many people in the background, walking backward"

    mod_value = pipeline.vae_scale_factor_spatial * pipeline.transformer.config.patch_size
    max_area = resolution[0] * resolution[1]
    for filename in filenames[:args['test_num']]:
        image = load_image(os.path.join(input_image_path, filename))
        aspect_ratio = image.height / image.width
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize(resolution)
        prompt = args['prompt']
        video = pipeline(image=image, prompt=prompt, guidance_scale=args['guidance_scale'], num_frames=49,
                      num_inference_steps=args['steps'], height=height, width=width,
                      negative_prompt=negative_prompt).frames[0]

        export_to_video(video, os.path.join(save_video_path,f"{filename.split('.')[0]}.mp4")
                        , fps=args['fps'], quality=args['quality'])
    gc.collect()
    torch.cuda.empty_cache()

@torch.no_grad
def test_I2VGen(input_image_path,out_video_path,resolution=(720,480),**args):

    pipe = I2VGenXLPipeline.from_pretrained(
        "ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16"
    )
    pipe.to(device=args['device'])
    pipe.enable_model_cpu_offload()
    save_video_path = os.path.join(out_video_path,'I2VGen')
    os.makedirs(save_video_path, exist_ok=True)
    filenames = os.listdir(input_image_path)
    mod_value = pipe.vae_scale_factor
    max_area = resolution[0] * resolution[1]
    negative_prompt = "Vibrant colors, overexposed, static, blurry details, subtitles, style, artwork, painting, scene, static, overall grayish, poor quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, deformed limbs, fused fingers, static scene, cluttered background, three legs, many people in the background, walking backward"
    for filename in filenames[:args['test_num']]:
        image = load_image(os.path.join(input_image_path, filename))
        aspect_ratio = image.height / image.width
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize(resolution)
        prompt = args['prompt']
        video = pipe(image=image, prompt=prompt, guidance_scale=args['guidance_scale'], num_frames=49,
                      num_inference_steps=args['steps'], height=height, width=width, negative_prompt=negative_prompt).frames[0]

        export_to_video(video, os.path.join(save_video_path,f"{filename.split('.')[0]}.mp4")
                        , fps=args['fps'], quality=args['quality'])
    gc.collect()
    torch.cuda.empty_cache()

@torch.no_grad
def test_SVD(input_image_path,out_video_path,resolution=(720,480),**args):
    # https://github.com/Stability-AI/generative-models
    pipe = DiffusionPipeline.from_pretrained(
  "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
    pipe.to(args['device'])
    pipe.enable_model_cpu_offload()
    save_video_path = os.path.join(out_video_path,'SVD')
    os.makedirs(save_video_path, exist_ok=True)
    filenames = os.listdir(input_image_path)
    generator = torch.Generator(device=args['device']).manual_seed(0)
    mod_value = pipe.vae_scale_factor
    max_area = resolution[0] * resolution[1]
    for filename in filenames[:args['test_num']]:
        image = load_image(os.path.join(input_image_path, filename))
        aspect_ratio = image.height / image.width
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize(resolution)
        prompt = args['prompt']
        video = pipe(image=image, num_frames=49,
                      num_inference_steps=args['steps'], height=height, width=width, generator=generator,
                      ).frames[0]

        export_to_video(video, os.path.join(save_video_path,f"{filename.split('.')[0]}.mp4")
                        , fps=args['fps'], quality=args['quality'])
    gc.collect()
    torch.cuda.empty_cache()

def choose_test(choice: list) -> list:
    '''
    choice : for example ["Wan", "CogVideoX"]
    return : [test_CogVideoX, test_Wan]
    '''
    return [globals().get(f"test_{name}") for name in choice if globals().get(f"test_{name}") is not None]

if __name__=="__main__":
    funcs = choose_test(["Wan", "CogVideoX"])
    for func in funcs:
        func()