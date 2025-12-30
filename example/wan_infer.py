from diffusers import WanPipeline
from diffusers.utils import export_to_video
import torch, os, gc
import torch.nn.functional as F
import argparse
from modify_model.modify_wan import set_sage_attn_wan
from tqdm import tqdm
from sageattention import sageattn
from contextlib import nullcontext

ATTENTION = {
    "sage": sageattn,
    "sdpa": F.scaled_dot_product_attention,
}

# os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")
os.environ["TOKENIZERS_PARALLELISM"]="false"
negative_prompt_1 = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
negative_prompt_2 = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
prompt_path = "videos/testing_prompts.txt"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--model", choices=["wan2.1-1.3b", "wan2.1-14b", "wan2.2-14b"], default="wan2.1-1.3b", help="Wan model")
    parser.add_argument('--compile', action='store_true', help='Compile the model')
    parser.add_argument('--attention_type', type=str, default='sage', choices=['sdpa', 'sage'], help='Attention type')
    parser.add_argument("--start", type=int, default=0, help="Starting prompt id of this run.")
    parser.add_argument("--end", type=int, default=12, help="Ending prompt id of this run.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    if args.model == "wan2.1-1.3b": model_path = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    elif args.model == "wan2.1-14b": model_path = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    else: model_path = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

    video_dir = f"videos/{args.model}/{args.attention_type}"
    os.makedirs(video_dir, exist_ok=True)

    with open(prompt_path, "r", encoding="utf-8") as file:
        prompts = file.readlines()
    selected_prompts = [p.strip() for p in prompts[args.start:args.end]]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=device).manual_seed(42)

    pipe = WanPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)

    set_sage_attn_wan(pipe.transformer, ATTENTION[args.attention_type])
    if getattr(pipe, "transformer_2", None) is not None: # Wan2.2
        set_sage_attn_wan(pipe.transformer_2, ATTENTION[args.attention_type])

    # if args.compile:
    #     pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")
    #     if getattr(pipe, "transformer_2", None) is not None: # Wan2.2
    #         pipe.transformer_2 = torch.compile(pipe.transformer_2, mode="max-autotune-no-cudagraphs")

    # pipe.enable_model_cpu_offload()
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    for local_i, prompt in tqdm(enumerate(selected_prompts), total=len(selected_prompts)):
        global_i = args.start + local_i
        amp_ctx = torch.autocast("cuda", torch.bfloat16, cache_enabled=False) if device == "cuda" else nullcontext()
        with amp_ctx:
            if args.model != "wan2.2-14b": # Wan2.1
                video = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt_1,
                    height=480,
                    width=832,
                    num_frames=81,
                    guidance_scale=5.0,
                    generator=gen,
                ).frames[0]
            else:
                video = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt_2,
                    height=720,
                    width=1280,
                    num_frames=81,
                    guidance_scale=4.0,
                    guidance_scale_2=3.0,
                    num_inference_steps=40,
                    generator=gen,
                ).frames[0]

            export_to_video(video, f"{video_dir}/{global_i}.mp4", fps=16)
            del video
            gc.collect()
            torch.cuda.empty_cache()
