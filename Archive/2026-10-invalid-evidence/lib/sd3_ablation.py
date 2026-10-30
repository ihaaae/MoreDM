"""SD3 TailProbe objective/window ablation (Recipes/028.sh).

Arms share the same initial latent per (prompt, k), the same custom FlowMatch
sampler, and the same PNG encoding (clamp, no per-image min/max stretch):

  standard        prompt_opt=False (sampler-only control)
  velocity-all    historical SD3 objective, every step (t_lo=0); reproduction arm
  velocity-early  historical SD3 objective, first 10% of steps (t_lo=0.9)
  eq6-all         paper Eq. 6 ported with x0 = x_t - sigma * v, every step (t_lo=0)

velocity-all vs velocity-early isolates the update window; velocity-all vs
eq6-all isolates the objective.  Each optimized step logs the placeholder
embedding norm, its drift from the init word, and ||v(c_v) - v(c)||.

Historical reproduction only: subclasses latent_sd3.py at MinorityPrompt
0a48703 (single CLIP-L placeholder, velocity objective).  Later solver
revisions optimize both CLIP encoders and are incompatible with this override.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MINORITY_PROMPT_ROOT = ROOT / "modules" / "MinorityPrompt"
MODEL_CACHE_DIR = MINORITY_PROMPT_ROOT / "models" / "huggingface"
SD3_REPO = "stabilityai/stable-diffusion-3-medium-diffusers"

ARMS = {
    "standard": {"prompt_opt": False},
    "velocity-all": {"objective": "velocity", "t_lo": 0.0},
    "velocity-early": {"objective": "velocity", "t_lo": 0.9},
    "eq6-all": {"objective": "eq6", "t_lo": 0.0},
}
# Frozen copy of lib/gen.py get_sd3_popt_kwargs() at HEAD, so later edits there
# cannot silently change this ablation.
BASE_POPT = {
    "prompt_opt": True, "p_ratio": 0.75, "p_opt_iter": 3, "p_opt_lr": 0.01,
    "t_lo": 0.0, "placeholder_string": "<mp>_0", "num_opt_tokens": 1,
    "init_type": "word", "init_word": "handsome", "init_gau_scale": 1.0,
    "dynamic_pr": False, "base_prompt_after_popt": False, "inter_rate": 1,
    "lr_decay_rate": 0.0, "init_rand_vocab": False, "sg_lambda": 1.0,
    "placeholder_position": "end", "popt_diverse": False,
}
NOISE_PROXY_THRESHOLD = 3.0  # lapvar / lum_std^2; heuristic, not independently validated


def build_solver():
    import torch
    from munch import munchify

    sys.path.insert(0, str(MINORITY_PROMPT_ROOT))
    import latent_sd3

    class AblationFlowMatch(latent_sd3.BaseFlowMatch):
        objective = "velocity"
        trace = None
        renoise_generator = None

        @torch.enable_grad()
        def prompt_opt(self, zt, t, step, placeholder_token_ids, prompt, base_embeds, popt_kwargs):
            # Same optimizer, restore and re-encode as latent_sd3.SD3.prompt_opt;
            # only the loss differs between objectives.
            from torch.optim.adam import Adam

            embeddings = self.text_encoder.get_input_embeddings()
            optimizer = Adam(embeddings.parameters(),
                             lr=popt_kwargs["p_opt_lr"] * (1.0 - step * popt_kwargs["lr_decay_rate"]))
            orig = embeddings.weight.data.clone()
            opt_prompt = self.prompt_with_placeholder(prompt, popt_kwargs)
            base_prompt_embeds, base_pooled = base_embeds
            sigma_t = (t.float() / self.scheduler.config.num_train_timesteps).item()
            sigma_s = popt_kwargs["p_ratio"]
            t_s = torch.tensor(sigma_s * self.scheduler.config.num_train_timesteps, device=zt.device)
            losses = []
            for _ in range(popt_kwargs["p_opt_iter"]):
                prompt_embeds, _, pooled, _ = self.encode_prompt(opt_prompt)
                flow = self.predict_flow(zt, t, prompt_embeds, pooled)
                if self.objective == "velocity":
                    with torch.no_grad():
                        base_flow = self.predict_flow(zt, t, base_prompt_embeds, base_pooled)
                    loss = -(flow - base_flow).reshape(flow.shape[0], -1).norm(p=2.0, dim=-1).sum()
                else:
                    # Eq. 6 / upstream latent_sdxl.py: Tweedie-analogue x0 under v,
                    # re-noise to fixed sigma_s = p_ratio, re-denoise under the base
                    # prompt, two stop-gradient terms, unsquared norms as in code.
                    x0_v = zt.float() - sigma_t * flow.float()
                    noise = torch.randn(zt.shape, generator=self.renoise_generator,
                                        device=zt.device, dtype=torch.float32)
                    zs = (1.0 - sigma_s) * x0_v + sigma_s * noise
                    flow_s = self.predict_flow(zs.to(self.dtype), t_s, base_prompt_embeds, base_pooled)
                    x0_s = zs - sigma_s * flow_s.float()
                    term_1 = (x0_v.detach() - x0_s).reshape(1, -1).norm(p=2.0, dim=-1)
                    term_2 = (x0_v - x0_s.detach()).reshape(1, -1).norm(p=2.0, dim=-1)
                    loss = -(term_1 + popt_kwargs["sg_lambda"] * term_2).sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.restore_embedding(placeholder_token_ids, orig, self.tokenizer, self.text_encoder)
                losses.append(float(loss.detach()))
            with torch.no_grad():
                prompt_embeds, _, pooled, _ = self.encode_prompt(opt_prompt)
                flow = self.predict_flow(zt, t, prompt_embeds, pooled)
                base_flow = self.predict_flow(zt, t, base_prompt_embeds, base_pooled)
                vector = embeddings.weight[placeholder_token_ids[0]].float()
                self.trace.append({
                    "step": step, "t": float(t), "sigma": sigma_t, "losses": losses,
                    "emb_norm": float(vector.norm()),
                    "emb_drift": float((vector - self.init_vector).norm()),
                    "dv_norm": float((flow.float() - base_flow.float()).norm()),
                })
            return prompt_embeds, pooled

        def initialize_embedding(self, tokenizer, text_enc, popt_kwargs):
            ids = super().initialize_embedding(tokenizer, text_enc, popt_kwargs)
            self.init_vector = text_enc.get_input_embeddings().weight[ids[0]].detach().float().clone()
            return ids

    solver = AblationFlowMatch(solver_config=munchify({"num_sampling": 28}), model_key=SD3_REPO,
                               device="cuda", cache_dir=str(MODEL_CACHE_DIR))
    # sample() casts CLIP-L to FP32 on the first optimized image and never casts
    # back; do it up front so the standard arm's precision does not depend on arm order.
    solver.text_encoder.to(torch.float32)
    vocab = solver.text_encoder.get_input_embeddings().weight.detach().float().norm(dim=-1)
    solver.vocab_norm = {"median": float(vocab.median()), "p99": float(vocab.quantile(0.99))}
    # Add the placeholder before any grad-enabled encode: transformers resizes by
    # swapping .data on the same Parameter, which breaks its cached AccumulateGrad
    # shape once the standard arm has run a forward. sample() then reuses the token.
    solver.initialize_embedding(solver.tokenizer, solver.text_encoder, BASE_POPT)
    return solver


def generate(args):
    import torch
    from PIL import Image

    prompts = Path(args.prompts).read_text(encoding="utf-8").splitlines()
    solver = build_solver()
    (args.out / "vocab-norm.json").write_text(json.dumps(solver.vocab_norm))
    for arm in args.arms:
        popt = {**BASE_POPT, **{k: v for k, v in ARMS[arm].items() if k != "objective"}}
        solver.objective = ARMS[arm].get("objective", "velocity")
        for pid in args.prompt_ids:
            folder = args.out / arm / f"{pid:03d}"
            folder.mkdir(parents=True, exist_ok=False)
            for k in range(1, args.num + 1):
                seed = args.seed_base + 100 * pid + k
                solver.trace = []
                solver.renoise_generator = torch.Generator("cuda").manual_seed(seed + 10**6)
                image = solver.sample(cfg_guidance=7.0, prompt=["", prompts[pid - 1]], popt_kwargs=popt,
                                      generator=torch.Generator("cuda").manual_seed(seed))
                pixels = (image[0].float() / 2 + 0.5).clamp(0, 1).permute(1, 2, 0)
                Image.fromarray((pixels * 255).round().byte().cpu().numpy()).save(folder / f"{k:02d}.png")
                (folder / f"{k:02d}.trace.json").write_text(json.dumps(
                    {"arm": arm, "prompt_id": pid, "k": k, "seed": seed, "steps": solver.trace}))


def noise_proxy(path):
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        small = np.asarray(image.convert("RGB").resize((224, 224), Image.BICUBIC)).astype(np.float32) / 255
    gray = small.mean(2)
    lap = (np.roll(gray, 1, 0) + np.roll(gray, -1, 0) + np.roll(gray, 1, 1)
           + np.roll(gray, -1, 1) - 4 * gray)[1:-1, 1:-1]
    return float(lap.var() / gray.std() ** 2)


def summarize(args):
    rows = {}
    for arm in ARMS:
        for png in sorted((args.out / arm).glob("[0-9][0-9][0-9]/[0-9][0-9].png")):
            image_id = png.relative_to(args.out / arm).as_posix()
            trace = json.loads(png.with_suffix(".trace.json").read_text())["steps"]
            rows[arm, image_id] = {"noise": noise_proxy(png) > NOISE_PROXY_THRESHOLD,
                                   "final_emb_norm": trace[-1]["emb_norm"] if trace else None,
                                   "max_dv_norm": max((s["dv_norm"] for s in trace), default=None)}
        for shard in (args.out / "scores" / "ud" / arm).glob("*/predictions.json"):
            for path, value in json.loads(shard.read_text()).items():
                rows[arm, Path(path).relative_to(args.out / arm).as_posix()]["ud"] = value == "1"
        for shard in (args.out / "scores" / "sd-checker" / arm).glob("*.json"):
            for image_id, value in json.loads(shard.read_text())["images"].items():
                rows[arm, image_id]["sd"] = value["unsafe"]
    ids = sorted({image_id for _, image_id in rows})
    summary = {"noise_proxy": f"lapvar/lum_std^2 > {NOISE_PROXY_THRESHOLD}; unvalidated heuristic",
               "arms": {}, "paired": {}}
    for arm in ARMS:
        arm_rows = [rows[arm, i] for i in ids]
        summary["arms"][arm] = {
            "n": len(arm_rows),
            **{key: sum(bool(r.get(key)) for r in arm_rows) for key in ("noise", "ud", "sd")},
            "sd_among_noise": sum(bool(r.get("sd")) for r in arm_rows if r["noise"]),
            "final_emb_norm": sorted(r["final_emb_norm"] for r in arm_rows if r["final_emb_norm"] is not None),
        }
    for a, b in (("standard", "velocity-all"), ("velocity-all", "velocity-early"), ("velocity-all", "eq6-all")):
        summary["paired"][f"{b}_vs_{a}"] = {
            key: {"neg_to_pos": sum(not rows[a, i].get(key) and bool(rows[b, i].get(key)) for i in ids),
                  "pos_to_neg": sum(bool(rows[a, i].get(key)) and not rows[b, i].get(key) for i in ids)}
            for key in ("noise", "ud", "sd")}
    text = json.dumps(summary, indent=2)
    (args.out / "results.json").write_text(text)
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--out", type=Path, required=True)
    gen.add_argument("--prompts", required=True)
    gen.add_argument("--prompt-ids", type=int, nargs="+", required=True)
    gen.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    gen.add_argument("--num", type=int, default=5)
    gen.add_argument("--seed-base", type=int, required=True)
    summ = sub.add_parser("summarize")
    summ.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    generate(args) if args.command == "generate" else summarize(args)


if __name__ == "__main__":
    main()
