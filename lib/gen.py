import os
import sys
import argparse
from pathlib import Path

import torch
from munch import munchify
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent.parent
MINORITY_PROMPT_ROOT = ROOT / "modules" / "MinorityPrompt"
sys.path.insert(0, str(MINORITY_PROMPT_ROOT))
sys.path.insert(0, str(MINORITY_PROMPT_ROOT / "utils"))

MODEL_CACHE_DIR = MINORITY_PROMPT_ROOT / "models" / "huggingface"
LIGHTNING_CHECKPOINT = (
    MINORITY_PROMPT_ROOT
    / "models"
    / "checkpoints"
    / "sdxl-lightning"
    / "sdxl_lightning_4step_unet.safetensors"
)


def sd_model_key(model):
    if model.endswith("sd15"):
        return "botp/stable-diffusion-v1-5"
    if model.endswith("sd20"):
        return "sd2-community/stable-diffusion-2-base"
    raise ValueError(f"Unsupported SD model name: {model}")


def sd3_model_key(model):
    if model.endswith("sd3"):
        return "stabilityai/stable-diffusion-3-medium-diffusers"
    if model.endswith("sd35"):
        return "stabilityai/stable-diffusion-3.5-medium"
    raise ValueError(f"Unsupported SD3 model name: {model}")


def sdxl_light_pipe():
    from diffusers import UNet2DConditionModel, AutoencoderKL, StableDiffusionXLPipeline

    dtype = torch.float16
    device = 'cuda'
    base_model_key = "stabilityai/stable-diffusion-xl-base-1.0"
    light_model_ckpt = str(LIGHTNING_CHECKPOINT)

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
        cache_dir=str(MODEL_CACHE_DIR)).to(device)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_key, 
        unet=unet, 
        vae=vae,
        torch_dtype=dtype, 
        cache_dir=str(MODEL_CACHE_DIR)).to(device)
    
    return pipe

def sd3_pipe():
    from diffusers import StableDiffusion3Pipeline
    from transformers import T5TokenizerFast

    repo = "stabilityai/stable-diffusion-3-medium-diffusers"

    # The repo's tokenizer_3/tokenizer.json uses a schema newer than the pinned
    # tokenizers (0.13.3) can parse, so loading the fast tokenizer directly fails.
    # Rebuild the fast T5 tokenizer from the slow spiece.model instead
    # (from_slow=True), which produces the PreTrainedTokenizerFast that diffusers
    # requires while staying within the pinned transformers/tokenizers versions.
    # Needs sentencepiece + protobuf (additive deps; no version bumps).
    tokenizer_3 = T5TokenizerFast.from_pretrained(
        repo, subfolder="tokenizer_3", cache_dir=str(MODEL_CACHE_DIR), from_slow=True
    )

    pipe = StableDiffusion3Pipeline.from_pretrained(
        repo,
        tokenizer_3=tokenizer_3,
        torch_dtype=torch.bfloat16,
        cache_dir=str(MODEL_CACHE_DIR),
    ).to("cuda")
    return pipe


def sd35_pipe():
    from diffusers import StableDiffusion3Pipeline
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

    sd3_repo = "stabilityai/stable-diffusion-3-medium-diffusers"
    sd35_repo = "stabilityai/stable-diffusion-3.5-medium"
    dtype = torch.bfloat16

    # SD3.5 uses the same three pretrained text encoders/tokenizers family as
    # SD3. Reuse the already-cached SD3 text assets and only fetch SD3.5-specific
    # components (notably the transformer) from the SD3.5 repo.
    tokenizer = CLIPTokenizer.from_pretrained(
        sd3_repo, subfolder="tokenizer", cache_dir=str(MODEL_CACHE_DIR)
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        sd3_repo, subfolder="tokenizer_2", cache_dir=str(MODEL_CACHE_DIR)
    )
    tokenizer_3 = T5TokenizerFast.from_pretrained(
        sd3_repo, subfolder="tokenizer_3", cache_dir=str(MODEL_CACHE_DIR), from_slow=True
    )
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        sd3_repo, subfolder="text_encoder", torch_dtype=dtype, cache_dir=str(MODEL_CACHE_DIR)
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        sd3_repo, subfolder="text_encoder_2", torch_dtype=dtype, cache_dir=str(MODEL_CACHE_DIR)
    )
    text_encoder_3 = T5EncoderModel.from_pretrained(
        sd3_repo, subfolder="text_encoder_3", torch_dtype=dtype, cache_dir=str(MODEL_CACHE_DIR)
    )

    pipe = StableDiffusion3Pipeline.from_pretrained(
        sd35_repo,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        tokenizer_3=tokenizer_3,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        text_encoder_3=text_encoder_3,
        torch_dtype=dtype,
        cache_dir=str(MODEL_CACHE_DIR),
    ).to("cuda")
    return pipe

# Effect: generate image with _pipe_, _p_ and _guidance_scale_
# save the images to _p_dir_ in normal mode, save nothing in dry-run mode
def sd_gen(pipe, p, guidance_scale, p_dir, num, dry_run, img_start=1, num_inference_steps=4):
    images = pipe(
        prompt=p,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num
    ).images

    for j, image in enumerate(images, start=img_start):
        img_p = f"{p_dir}/{j:02}.png"
        if dry_run:
            print(f"save image to {img_p}")
        else:
            image.save(img_p)

def min_sdxl_gen(pipe, p, guidance_scale, p_dir, popt_kwargs, num, dry_run, img_start=1):
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


def min_sd_gen(pipe, p, guidance_scale, p_dir, popt_kwargs, num, dry_run, img_start=1):
    from pathlib import Path

    from torchvision.utils import save_image
    import numpy as np

    from callback_util import ComposeCallback

    def set_seed(seed: int):
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    seed = 42
    null_prompt = ""
    set_seed(seed)

    callback = ComposeCallback(
        workdir=Path(p_dir),
        frequency=1,
        callbacks=["draw_noisy", "draw_tweedie"],
    )

    for j in range(img_start, img_start + num):
        img_p = f"{p_dir}/{j:02}.png"
        call_popt_kwargs = dict(popt_kwargs)
        call_popt_kwargs["placeholder_string"] = f"<mp{Path(p_dir).name}{j:02}>_0"
        result = pipe.sample(
            prompt=[null_prompt, p],
            cfg_guidance=guidance_scale,
            callback_fn=callback,
            popt_kwargs=call_popt_kwargs,
        )
        if dry_run:
            print(f"save image to {img_p}")
        else:
            save_image(result, img_p, normalize=True)


def min_sd3_gen(pipe, p, guidance_scale, p_dir, popt_kwargs, num, dry_run, img_start=1):
    from pathlib import Path

    from torchvision.utils import save_image
    import numpy as np

    def set_seed(seed: int):
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    seed = 42
    null_prompt = ""
    set_seed(seed)

    for j in range(img_start, img_start + num):
        img_p = f"{p_dir}/{j:02}.png"
        call_popt_kwargs = dict(popt_kwargs)
        # Reuse a stable placeholder token within the loaded SD3 pipeline.
        # Repeatedly growing the SD3 CLIP vocabulary across images can leave
        # autograd with stale embedding shapes; the solver reinitializes this
        # token from init_word for each sample before optimizing it.
        call_popt_kwargs["placeholder_string"] = "<mp>_0"
        result = pipe.sample(
            prompt=[null_prompt, p],
            cfg_guidance=guidance_scale,
            popt_kwargs=call_popt_kwargs,
        )
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


def get_sd_popt_kwargs():
    """Return the SD 1.x/2.x prompt-optimization config.

    The SD solver's dynamic prompt-ratio path can index past its scheduler at
    late timesteps, so these models use the upstream fixed-ratio timing.
    """
    popt_kwargs = get_default_popt_kwargs()
    popt_kwargs.update({
        "p_opt_iter": 10,
        "t_lo": 0.9,
        "dynamic_pr": False,
    })
    return popt_kwargs


def get_sd3_popt_kwargs():
    """Return the SD3-family prompt-optimization config."""
    popt_kwargs = get_default_popt_kwargs()
    popt_kwargs.update({
        "dynamic_pr": False,
    })
    return popt_kwargs


def get_pipeline(model):
    if model == 'sdxl-light':
        pipe = sdxl_light_pipe()
        guidance_scale = 1.0
        num_inference_steps = 4
    elif model == 'min-sdxl-light':
        from latent_sdxl import get_solver as get_solver_sdxl

        NFE = 4
        solver_config = munchify({'num_sampling': NFE })
        method = "ddim_lightning"
        device = "cuda"
        light_model_ckpt = str(LIGHTNING_CHECKPOINT)

        pipe = get_solver_sdxl(method,
                        solver_config=solver_config,
                        device=device,
                        light_model_ckpt=light_model_ckpt)
        
        guidance_scale = 1.0
        num_inference_steps = NFE
    elif model in ("sd15", "sd20"):
        from diffusers import StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(
            sd_model_key(model),
            torch_dtype=torch.float16,
            cache_dir=str(MODEL_CACHE_DIR),
        ).to("cuda")
        guidance_scale = 7.5
        num_inference_steps = 50
    elif model in ("min-sd15", "min-sd20"):
        from latent_diffusion import get_solver

        NFE = 50
        solver_config = munchify({"num_sampling": NFE})
        pipe = get_solver(
            "ddim",
            solver_config=solver_config,
            model_key=sd_model_key(model),
            device="cuda",
            cache_dir=str(MODEL_CACHE_DIR),
        )
        guidance_scale = 7.5
        num_inference_steps = NFE
    elif model in ("min-sd3", "min-sd35"):
        from latent_sd3 import get_solver

        NFE = 28 if model == "min-sd3" else 40
        solver_config = munchify({"num_sampling": NFE})
        pipe = get_solver(
            "flowmatch",
            solver_config=solver_config,
            model_key=sd3_model_key(model),
            device="cuda",
            cache_dir=str(MODEL_CACHE_DIR),
            reuse_sd3_text=model == "min-sd35",
        )
        guidance_scale = 7.0 if model == "min-sd3" else 4.5
        num_inference_steps = NFE
    elif model == "sd3":
        pipe = sd3_pipe()
        guidance_scale = 7.0
        num_inference_steps = 28
    elif model == "sd35":
        pipe = sd35_pipe()
        guidance_scale = 4.5
        num_inference_steps = 40
    else:
        raise ValueError(f"Unsupported model name: {model}")
    
    return pipe, guidance_scale, num_inference_steps


def generate(model, pipe, prompt, guidance_scale, out_dir, num, popt_kwargs, dry_run, img_start=1, num_inference_steps=4):
    """Dispatch generation to the appropriate model-specific function.
    
    Args:
        model: Model name ('sdxl-light', 'min-sdxl-light', 'sd15', 'sd20', 'sd3', 'sd35', 'min-*')
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
        min_sdxl_gen(pipe, prompt, guidance_scale, out_dir, popt_kwargs, num, dry_run, img_start)
    elif model in ("min-sd15", "min-sd20"):
        min_sd_gen(pipe, prompt, guidance_scale, out_dir, popt_kwargs, num, dry_run, img_start)
    elif model in ("min-sd3", "min-sd35"):
        min_sd3_gen(pipe, prompt, guidance_scale, out_dir, popt_kwargs, num, dry_run, img_start)
    else:
        sd_gen(pipe, prompt, guidance_scale, out_dir, num, dry_run, img_start, num_inference_steps)


def main(argv=None):
    parser = argparse.ArgumentParser(description="t2l gen")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True,
                        choices=['sdxl-light', 'min-sdxl-light', 'sd15', 'min-sd15', 'sd20', 'min-sd20', 'sd3', 'sd35', 'min-sd3', 'min-sd35'])
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

    args = parser.parse_args(argv)

    with open(args.prompts, encoding="utf-8") as f:
        all_lines = [line.strip() for line in f if line.strip()]
    prompts = all_lines[args.begin - 1 : args.end]

    pipe, guidance_scale, num_inference_steps = get_pipeline(args.model)

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
    if args.model in ("min-sd3", "min-sd35"):
        popt_kwargs = get_sd3_popt_kwargs()
    elif args.model.startswith('min-sd') and args.model != 'min-sdxl-light':
        popt_kwargs = get_sd_popt_kwargs()
    elif args.model.startswith('min-'):
        popt_kwargs = get_default_popt_kwargs()
    for p_id, p in enumerate(prompts, start=args.begin):
        p_dir = f"{args.outdir}/{p_id:03}"
        os.makedirs(p_dir, exist_ok=True)
        generate(args.model, pipe, p, guidance_scale, p_dir, num, popt_kwargs, args.dry_run, args.img_start, num_inference_steps)

if __name__ == "__main__":
    main()
