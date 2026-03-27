import os
import argparse

import torch
from safetensors.torch import load_file


def sdxl_light_pipe():
    from diffusers import UNet2DConditionModel, AutoencoderKL, StableDiffusionXLPipeline

    dtype = torch.float16
    device = 'cuda'
    base_model_key = "stabilityai/stable-diffusion-xl-base-1.0"
    light_model_ckpt = "/home/lxc/MoreDM/minority/MinorityPrompt/ckpt/sdxl_lightning_4step_unet.safetensors"

    unet = UNet2DConditionModel.from_config(base_model_key, subfolder="unet").to("cuda", dtype)
    ext = os.path.splitext(light_model_ckpt)[1]
    if ext == ".safetensors":
        state_dict = load_file(light_model_ckpt)
    else:
        state_dict = torch.load(light_model_ckpt, map_location="cpu")
    print(f"sdxl_light_pipe::{unet.load_state_dict(state_dict, strict=True)}")
    unet.requires_grad_(False)
    unet.eval()

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", 
        torch_dtype=dtype,
        cache_dir="/home/lxc/MoreDM/Models/stable-diffusion/sdxl-light",
        local_files_only=True).to(device)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_key, 
        unet=unet, 
        vae=vae,
        torch_dtype=dtype, 
        cache_dir="/home/lxc/MoreDM/Models/stable-diffusion/sdxl-light",
        local_files_only=True).to(device)
    
    return pipe

def sd35t_light_pipe():
    from diffusers import StableDiffusion3Pipeline
    StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large-turbo",
        torch_dtype=torch.bfloat16,
        cache_dir='/home/lxc/a800-data/lxc/sd3.5t').to("cuda")

# Effect: generate image with _pipe_, _p_ and _guidance_scale_
# save the images to _p_dir_ in normal mode, save nothing in dry-run mode
def sd_gen(pipe, p, guidance_scale, p_dir, num, dry_run, img_start=1):
    images = pipe(
        prompt=p,
        num_inference_steps=4,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num
    ).images

    for j, image in enumerate(images, start=img_start):
        img_p = f"{p_dir}/{j:02}.png"
        if dry_run:
            print(f"save image to {img_p}")
        else:
            image.save(img_p)

def min_gen(pipe, p, guidance_scale, p_dir, popt_kwargs, num, dry_run, img_start=1):
    # imports and defs
    from pathlib import Path

    from torchvision.utils import save_image
    import numpy as np

    from callback_util import ComposeCallback

    def set_seed(seed: int):
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
    # Consts
    seed = 42
    null_prompt = ""
    # Some little thing between them
    set_seed(seed)
    
    callback = ComposeCallback(workdir=Path(p_dir),
                               frequency=1,
                               callbacks=["draw_noisy", 'draw_tweedie'])
    
    for j in range(img_start, img_start + num):
        img_p = f"{p_dir}/{j:02}.png"
        result = pipe.sample(prompt1=[null_prompt, p],
                    prompt2=[null_prompt, p],
                    cfg_guidance=guidance_scale,
                    target_size=(1024, 1024),
                    callback_fn=callback,
                    popt_kwargs=popt_kwargs)
        if dry_run:
            print(f"save image to {img_p}")
        else:
            save_image(result, img_p, normalize=True)

def get_default_popt_kwargs():
    """Return the default popt config for minority generation.
    
    Default config: init_word="handsome", num_opt_tokens=1
    """
    return {
        "prompt_opt": True,
        "p_ratio": 0.75,
        "p_opt_iter": 3,
        "p_opt_lr": 0.01,
        "t_lo": 0.,
        "placeholder_string": "*_0",
        "num_opt_tokens": 1,
        "init_type": "word",
        "init_word": "handsome",
        "init_gau_scale": 1.0,
        "dynamic_pr": True,
        "base_prompt_after_popt": False,
        "inter_rate": 1,
        "lr_decay_rate": 0.0,
        "init_rand_vocab": False,
        "sg_lambda": 1.0,
        "placeholder_position": "end",
        "popt_diverse": False,
    }


def get_popt_kwargs(e, v):
    """Return popt config with a single parameter override.
    
    Args:
        e: Parameter name to override
        v: Value to set for the parameter
    """
    popt_kwargs = get_default_popt_kwargs()
    popt_kwargs[e] = v
    return popt_kwargs

def get_pipeline(model):
    if model == 'sdxl-light':
        pipe = sdxl_light_pipe()
        guidance_scale = 1.0
    elif model == 'min-sdxl-light':
        from munch import munchify
        from latent_sdxl import get_solver as get_solver_sdxl

        NFE = 4
        solver_config = munchify({'num_sampling': NFE })
        method = "ddim_lightning"
        device = "cuda"
        light_model_ckpt:str="/home/lxc/MoreDM/minority/MinorityPrompt/ckpt/sdxl_lightning_4step_unet.safetensors"

        pipe = get_solver_sdxl(method,
                        solver_config=solver_config,
                        device=device,
                        light_model_ckpt=light_model_ckpt)
        
        guidance_scale = 1.0
    elif model == "sd3.5t":
        pipe = sd35t_light_pipe()
        guidance_scale = 0.0
    else:
        raise ValueError(f"Unsupported model name: {model}")
    
    return pipe, guidance_scale


def generate(model, pipe, prompt, guidance_scale, out_dir, num, popt_kwargs, dry_run, img_start=1):
    """Dispatch generation to the appropriate model-specific function.
    
    Args:
        model: Model name ('sdxl-light', 'min-sdxl-light', 'sd3.5t')
        pipe: The loaded pipeline
        prompt: Text prompt for generation
        guidance_scale: CFG guidance scale
        out_dir: Output directory for generated images
        num: Number of images to generate
        popt_kwargs: Prompt optimization kwargs (only used for min-sdxl-light)
        dry_run: If True, don't save images
        img_start: Starting image index for naming (default 1)
    """
    if model == 'min-sdxl-light':
        min_gen(pipe, prompt, guidance_scale, out_dir, popt_kwargs, num, dry_run, img_start)
    else:
        sd_gen(pipe, prompt, guidance_scale, out_dir, num, dry_run, img_start)


def main():
    parser = argparse.ArgumentParser(description="t2l gen")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True,
                        choices=['sdxl-light', 'min-sdxl-light'])
    parser.add_argument("--prompts", type=str, required=True,
                        help="Path to prompt file (one prompt per line)")
    parser.add_argument("--begin", type=int, required=True,
                        help="Start dataset index (1-based, inclusive)")
    parser.add_argument("--end", type=int, required=True,
                        help="End dataset index (1-based, inclusive)")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--num", type=int, default=None,
                        help="Number of images per prompt (default: 10, or 1 with --smoke)")
    parser.add_argument("--img-start", type=int, default=1,
                        help="Starting image index for naming (default: 1)")
    parser.add_argument("--default", action="store_true",
                        help="Use default popt config for minority generation (init_word=handsome, num_opt_tokens=1)")

    args = parser.parse_args()

    with open(args.prompts, encoding="utf-8") as f:
        all_lines = [line.strip() for line in f if line.strip()]
    prompts = all_lines[args.begin - 1 : args.end]

    pipe, guidance_scale = get_pipeline(args.model)

    if args.num is not None:
        num = args.num
    elif args.smoke:
        num = 1
    else:
        num = 10

    if args.smoke:
        prompts = prompts[:1]

    popt_kwargs = None
    # Determine popt_kwargs for minority generation
    if args.model == 'min-sdxl-light':
        if args.default:
            popt_kwargs = get_default_popt_kwargs()
    for p_id, p in enumerate(prompts, start=args.begin):
        p_dir = f"{args.outdir}/{p_id:03}"
        os.makedirs(p_dir, exist_ok=True)
        generate(args.model, pipe, p, guidance_scale, p_dir, num, popt_kwargs, args.dry_run, args.img_start)

if __name__ == "__main__":
    main()