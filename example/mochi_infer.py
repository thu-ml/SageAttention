import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import gc
import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
import torch.nn.functional as F
from diffusers import MochiPipeline
from diffusers.utils import export_to_video
from tqdm import tqdm
from modify_model.modify_mochi import set_sage_attn_mochi
from sageattention import sageattn
from sageattention.fa3_wrapper import fa3
from sageattention.fa3_wrapper import fa3_fp8

ATTENTION = {
    "sdpa": F.scaled_dot_product_attention,
    "sage": sageattn,
    "fa3": fa3,
    "fa3_fp8": fa3_fp8
}

prompt_path = "videos/testing_prompts.txt"

def parse_args():
    parser = argparse.ArgumentParser(description="Mochi 1 Inference")
    parser.add_argument("--model_path", default="genmo/mochi-1-preview", help="Mochi model path")
    parser.add_argument("--compile", action="store_true", help="Compile the model")
    parser.add_argument(
        "--attention_type",
        type=str,
        default="sdpa",
        choices=["sdpa", "sage", "fa3", "fa3_fp8"],
        help="Attention type",
    )
    parser.add_argument("--start", type=int, default=0, help="Starting prompt id of this run.")
    parser.add_argument("--end", type=int, default=12, help="Ending prompt id of this run.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    video_dir = f"videos/mochi/{args.attention_type}"
    os.makedirs(video_dir, exist_ok=True)

    with open(prompt_path, "r", encoding="utf-8") as file:
        prompts = file.readlines()
    selected_prompts = [p.strip() for p in prompts[args.start : args.end]]

    pipe = MochiPipeline.from_pretrained(args.model_path, variant="bf16",torch_dtype=torch.bfloat16)

    set_sage_attn_mochi(pipe.transformer, ATTENTION[args.attention_type])

    if args.compile:
        pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

    # Enable memory savings
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()
    pipe.enable_vae_slicing()

    with torch.no_grad():
        for local_i, prompt in tqdm(enumerate(selected_prompts), total=len(selected_prompts)):
            global_i = args.start + local_i
            frames = pipe(
                prompt,
                height=480,
                width=848,
                num_frames=84, # can be changed to 163 for longer video
                guidance_scale=6.0,
                num_inference_steps=64,
                generator=torch.Generator(device="cuda").manual_seed(42),
            ).frames[0]
            export_to_video(frames, f"{video_dir}/{global_i}.mp4", fps=30)
            del frames
            gc.collect()
            torch.cuda.empty_cache()
