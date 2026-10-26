import os
import sys
import argparse
import hashlib
import importlib.metadata
import json
import time
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


def get_pipeline(model, sdxl_base_path=None):
    if model in ("sdxl", "min-sdxl"):
        if sdxl_base_path is None or not Path(sdxl_base_path).is_dir():
            raise ValueError("Base SDXL requires an explicit local --sdxl-base-path.")
        if model == "sdxl":
            from diffusers import AutoencoderKL, StableDiffusionXLPipeline

            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16,
                cache_dir=str(MODEL_CACHE_DIR), local_files_only=True,
            )
            pipe = StableDiffusionXLPipeline.from_pretrained(
                sdxl_base_path, vae=vae, torch_dtype=torch.float16,
                local_files_only=True,
            ).to("cuda")
        else:
            from latent_sdxl import get_solver as get_solver_sdxl

            pipe = get_solver_sdxl(
                "ddim", solver_config=munchify({"num_sampling": 50}),
                model_key=str(sdxl_base_path), device="cuda",
            )
        pipe.unet.requires_grad_(False)
        guidance_scale, num_inference_steps = 7.5, 50
    elif model == 'sdxl-light':
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


def audit_generate(args, prompts):
    """Generate fresh, seed-paired controls without changing legacy generation.

    Custom SDXL arms use the same upstream DDIM implementation per model.
    Diagnostic runs also support the original Diffusers pipeline, paired on
    CPU FP32 Gaussian noise before each sampler's native scaling/casting.
    The first UNet input is checked across custom arms, not across samplers.
    Phases occupy disjoint seed ranges; this is not a full recovery protocol.
    """
    from torchvision.utils import save_image

    if args.model not in ("min-sdxl-light", "sdxl-light", "sdxl", "min-sdxl", "sd15"):
        raise ValueError("Audit generation supports SDXL models and the sd15 reference.")
    if args.model in ("sd15", "sdxl-light", "sdxl") and args.audit_arms != ["standard"]:
        raise ValueError("Diffusers pipelines support only the standard arm.")
    if args.model in ("sdxl-light", "sdxl", "min-sdxl") and args.audit_phase != "diagnostic":
        raise ValueError("Base SDXL and the original pipeline require diagnostic runs.")
    if args.model in ("sdxl", "min-sdxl") and args.sdxl_base_path is None:
        raise ValueError("Base SDXL requires an explicit local --sdxl-base-path.")
    if args.audit_phase == "diagnostic" and args.model == "sd15":
        raise ValueError("The paired sampler diagnostic supports only SDXL-Lightning.")
    if args.audit_timestep_spacing is not None and (
        args.model != "sdxl-light" or args.audit_phase != "diagnostic"
    ):
        raise ValueError("Timestep-spacing overrides require the SDXL Diffusers diagnostic.")
    if args.audit_phase == "selection" and args.audit_arms != ["standard"]:
        raise ValueError("Selection must not inspect intervention outcomes.")
    if args.dry_run or args.smoke:
        raise ValueError("Use explicit --begin/--end/--num and --audit-phase pilot instead.")
    if args.seed_base is None or not 0 <= args.seed_base < 1_000_000_000:
        raise ValueError("Audit generation requires --seed-base in [0, 1000000000).")
    if not 1 <= args.begin <= args.end <= len(prompts) < 1_000_000:
        raise ValueError("Invalid inclusive prompt range.")
    if args.num is None or not 1 <= args.img_start <= args.img_start + args.num - 1 <= 999:
        raise ValueError("Set --num explicitly; image IDs must be in 1..999.")
    if len(set(args.audit_arms)) != len(args.audit_arms):
        raise ValueError("Audit arms must be unique.")
    prompt_ids = list(range(args.begin, args.end + 1))
    if args.prompt_ids is not None:
        if (len(set(args.prompt_ids)) != len(args.prompt_ids)
                or any(pid not in prompt_ids for pid in args.prompt_ids)):
            raise ValueError("Audit prompt IDs must be unique and inside --begin/--end.")
        prompt_ids = sorted(args.prompt_ids)

    def digest(path):
        checksum = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(chunk)
        return checksum.hexdigest()

    init_words = {}
    if args.audit_init_words:
        if args.model != "min-sdxl" or args.audit_arms != ["tailprobe"]:
            raise ValueError("Per-prompt initializers require min-sdxl tailprobe only.")
        selection = json.loads(Path(args.audit_init_words).read_text())
        for row in selection["selections"]:
            pid, word = row["prompt_id"], row["init_word"]
            if pid in init_words or not isinstance(word, str) or not word.strip():
                raise ValueError("Duplicate prompt ID or empty initializer.")
            init_words[pid] = word
        if any(pid not in init_words for pid in prompt_ids):
            raise ValueError("Missing per-prompt initializer.")

    outdir = Path(args.outdir)
    manifest = outdir / "manifests" / (
        f"{args.begin:03}-{args.end:03}-{args.img_start:02}-{args.num}.json"
    )
    if manifest.exists():
        raise FileExistsError(f"Existing shard manifest; choose a fresh output directory: {manifest}")
    for pid in prompt_ids:
        for iid in range(args.img_start, args.img_start + args.num):
            for arm in args.audit_arms:
                stem = outdir / arm / f"{pid:03}" / f"{iid:02}"
                if stem.with_suffix(".png").exists() or stem.with_suffix(".json").exists():
                    raise FileExistsError(f"Refusing to overwrite audit output: {stem}")

    configs = {}
    for arm in args.audit_arms:
        # Upstream non-Lightning defaults: 50 steps, CFG7.5, ten updates at
        # t>900, fixed p_ratio. Dynamic timing is invalid at the last DDIM step.
        config = get_sd_popt_kwargs() if args.model in ("sdxl", "min-sdxl") else get_default_popt_kwargs()
        if arm == "standard":
            config["prompt_opt"] = False
        elif arm in ("frozen", "random"):
            config["p_opt_iter"] = 0
            if arm == "random":
                # The upstream diagonal Gaussian matches vocabulary coordinate
                # means/variances. This is frozen random-v, not random updates
                # or a norm-matched adversarial control.
                config["init_type"] = "gaussian_white"
        configs[arm] = config

    phase_offset = {"pilot": 0, "selection": 1_000_000_000,
                    "evaluation": 2_000_000_000, "calibration": 3_000_000_000,
                    "diagnostic": 4_000_000_000}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": args.model,
        "phase": args.audit_phase,
        "seed_base": args.seed_base,
        "seed_rule": "seed_base + phase_offset + prompt_id * 1000 + image_id",
        "phase_offset": phase_offset[args.audit_phase],
        "begin": args.begin,
        "end": args.end,
        "prompt_ids": prompt_ids,
        "img_start": args.img_start,
        "num": args.num,
        "arms": args.audit_arms,
        "popt_kwargs": configs,
        "prompts_sha256": digest(args.prompts),
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in (
            Path(__file__), MINORITY_PROMPT_ROOT / "latent_sdxl.py", ROOT / "uv.lock"
        )},
        "package_versions": {name: importlib.metadata.version(name) for name in (
            "torch", "diffusers", "transformers", "torchvision"
        )},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "status": "started",
        "output_moderation": "disabled; detectors score saved raw outputs separately",
        "image_encoding": "clamp [0,1]; no per-image min/max normalization",
    }
    if init_words:
        metadata["init_words"] = {str(pid): init_words[pid] for pid in prompt_ids}
        metadata["init_words_sha256"] = digest(args.audit_init_words)
    if args.model in ("min-sdxl-light", "sdxl-light"):
        metadata["lightning_checkpoint_sha256"] = digest(LIGHTNING_CHECKPOINT)
    if args.model in ("sdxl", "min-sdxl"):
        metadata["base_unet_sha256"] = digest(Path(args.sdxl_base_path) / "unet/diffusion_pytorch_model.safetensors")
        metadata["base_model_path"] = str(Path(args.sdxl_base_path).resolve())
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)

    pipe, guidance_scale, num_steps = get_pipeline(args.model, args.sdxl_base_path)
    for pid in prompt_ids:
        if init_words and any(len(tokenizer.encode(init_words[pid], add_special_tokens=False)) != 1
                              for tokenizer in (pipe.tokenizer_1_base, pipe.tokenizer_2_base)):
            raise ValueError(f"Initializer for prompt {pid} must be one token in both tokenizers.")
    if args.audit_timestep_spacing is not None:
        pipe.scheduler = type(pipe.scheduler).from_config(
            pipe.scheduler.config, timestep_spacing=args.audit_timestep_spacing,
        )
    if args.model == "sd15":
        # Compare raw model behavior, not blacked-out moderation replacements.
        pipe.safety_checker = None
    denoiser = pipe.unet
    metadata.update({
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_steps,
        "scheduler_class": type(pipe.scheduler).__name__,
        "scheduler_config": dict(pipe.scheduler.config),
        "sampler": ("upstream ddim_lightning" if args.model == "min-sdxl-light" else
                    "upstream ddim" if args.model == "min-sdxl" else "diffusers pipeline"),
        "gpu": torch.cuda.get_device_name(),
    })
    if args.audit_phase == "diagnostic":
        metadata["noise_pairing"] = (
            "CPU torch.randn FP32 (1,4,128,128) with per-image seed; custom initializer "
            "verified byte-identical; Diffusers receives same noise cast to native FP16 "
            "before native init_noise_sigma scaling. No cross-sampler UNet-input equality."
        )
    manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for pid in prompt_ids:
        prompt = prompts[pid - 1]
        for iid in range(args.img_start, args.img_start + args.num):
            seed = args.seed_base + phase_offset[args.audit_phase] + pid * 1000 + iid
            paired_input = None
            base_noise = None
            if args.audit_phase == "diagnostic":
                base_noise = torch.randn(
                    (1, 4, 128, 128), dtype=torch.float32,
                    generator=torch.Generator(device="cpu").manual_seed(seed),
                )
            for arm in args.audit_arms:
                trace = {"unet_forward_calls": 0}
                original_initializer = None
                if base_noise is not None:
                    trace["base_noise_sha256"] = hashlib.sha256(base_noise.numpy().tobytes()).hexdigest()
                    if args.model in ("min-sdxl-light", "min-sdxl"):
                        original_initializer = pipe.initialize_latent

                        def checked_initializer(*positional, **keyword):
                            latent = original_initializer(*positional, **keyword)
                            if not torch.equal(latent.detach().cpu(), base_noise):
                                raise RuntimeError("Custom initializer differs from paired Gaussian noise")
                            trace["base_noise_verified"] = True
                            return latent

                        pipe.initialize_latent = checked_initializer

                def capture_input(module, positional, keyword):
                    trace["unet_forward_calls"] += 1
                    if "first_unet_input_sha256" not in trace:
                        latent = keyword.get("sample")
                        if latent is None:
                            latent = positional[0]
                        trace["first_unet_batch_size"] = latent.shape[0]
                        # The normal sampler batches unconditional/conditional
                        # copies; prompt optimization uses only the conditional
                        # pass. Compare the first latent, not the CFG batch.
                        latent = latent[:1].detach().contiguous().cpu()
                        trace["first_unet_input_sha256"] = hashlib.sha256(latent.numpy().tobytes()).hexdigest()
                        trace["first_unet_input_shape"] = list(latent.shape)
                        trace["first_unet_input_dtype"] = str(latent.dtype)

                hook = denoiser.register_forward_pre_hook(capture_input, with_kwargs=True)
                torch.manual_seed(seed)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                try:
                    with torch.no_grad():
                        if args.model == "sd15":
                            image = pipe(
                                prompt=prompt,
                                guidance_scale=guidance_scale,
                                num_inference_steps=num_steps,
                                generator=torch.Generator(device="cuda").manual_seed(seed),
                            ).images[0]
                        elif args.model in ("sdxl-light", "sdxl"):
                            latents = base_noise.to(device="cuda", dtype=pipe.unet.dtype)
                            trace["base_noise_verified"] = torch.equal(
                                latents.cpu(), base_noise.to(dtype=pipe.unet.dtype)
                            )
                            image = pipe(
                                prompt=prompt, guidance_scale=guidance_scale,
                                num_inference_steps=num_steps, latents=latents,
                                output_type="pt",
                            ).images
                        else:
                            sample_config = dict(configs[arm])
                            if init_words:
                                sample_config["init_word"] = init_words[pid]
                            image = pipe.sample(
                                prompt1=["", prompt], prompt2=["", prompt],
                                cfg_guidance=guidance_scale, target_size=(1024, 1024),
                                popt_kwargs=sample_config,
                            )
                    torch.cuda.synchronize()
                finally:
                    hook.remove()
                    if original_initializer is not None:
                        pipe.initialize_latent = original_initializer
                elapsed = time.perf_counter() - started
                if base_noise is not None and not trace.get("base_noise_verified"):
                    raise RuntimeError("Paired Gaussian noise was not verified")
                initial_input = trace["first_unet_input_sha256"]
                if paired_input is not None and initial_input != paired_input:
                    raise RuntimeError(f"Initial denoiser input mismatch at {pid}/{iid}, arm={arm}")
                paired_input = initial_input
                stem = outdir / arm / f"{pid:03}" / f"{iid:02}"
                stem.parent.mkdir(parents=True, exist_ok=True)
                if args.model == "sd15":
                    image.save(stem.with_suffix(".png"))
                else:
                    save_image(image, stem.with_suffix(".png"), normalize=False)
                record = {
                    "prompt_id": pid, "image_id": iid, "arm": arm, "seed": seed,
                    "phase": args.audit_phase, "seconds": elapsed,
                    "timing_warmup": pid == prompt_ids[0] and iid == args.img_start,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "image_sha256": digest(stem.with_suffix(".png")),
                    **trace,
                }
                if init_words:
                    record["init_word"] = init_words[pid]
                with stem.with_suffix(".json").open("x", encoding="utf-8") as stream:
                    json.dump(record, stream, indent=2)
    metadata["status"] = "complete"
    manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="t2l gen")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True,
                        choices=['sdxl-light', 'min-sdxl-light', 'sdxl', 'min-sdxl', 'sd15', 'min-sd15', 'sd20', 'min-sd20', 'sd3', 'sd35', 'min-sd3', 'min-sd35'])
    parser.add_argument("--sdxl-base-path", help="Local base SDXL snapshot for diagnostic runs; no Lightning weights.")
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
    parser.add_argument("--audit-arms", nargs="+", choices=["standard", "frozen", "random", "tailprobe"],
                        help="Fresh seed-paired audit arms; outputs go under <outdir>/<arm>.")
    parser.add_argument("--audit-phase", choices=["pilot", "selection", "evaluation", "calibration", "diagnostic"], default="pilot")
    parser.add_argument("--audit-timestep-spacing", choices=["leading", "trailing"],
                        help="SDXL Diffusers diagnostic only: override scheduler spacing, not other settings.")
    parser.add_argument("--prompt-ids", nargs="+", type=int,
                        help="Audit only: original prompt IDs to generate inside --begin/--end.")
    parser.add_argument("--seed-base", type=int,
                        help="Required for audit generation; shared across arms and shards.")
    parser.add_argument("--audit-init-words", help="Jev selections JSON; min-sdxl tailprobe only.")

    args = parser.parse_args(argv)
    if args.audit_init_words and not args.audit_arms:
        parser.error("--audit-init-words requires --audit-arms.")
    if args.model in ("sdxl", "min-sdxl") and not args.audit_arms:
        parser.error("Base SDXL is currently supported only through --audit-arms.")
    if args.sdxl_base_path is not None and args.model not in ("sdxl", "min-sdxl"):
        parser.error("--sdxl-base-path is only for sdxl/min-sdxl.")

    with open(args.prompts, encoding="utf-8") as f:
        lines = f.read().splitlines()
    if args.audit_arms:
        if any(not line.strip() for line in lines):
            parser.error("Audit prompt files must not contain blank lines (IDs must stay stable).")
        audit_generate(args, [line.strip() for line in lines])
        return
    if args.seed_base is not None or args.prompt_ids is not None or args.audit_timestep_spacing is not None:
        parser.error("Audit options require --audit-arms.")
    all_lines = [line.strip() for line in lines if line.strip()]
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
