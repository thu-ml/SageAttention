import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch, argparse, gc
from tqdm import tqdm
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from sageattention import sageattn
import torch.nn.functional as F

prompt_path = "videos/testing_prompts.txt"

def parse_args():
    parser = argparse.ArgumentParser(description="CogVideoX Inference")
    parser.add_argument("--model",choices=["cogvideox-2b", "cogvideox1.5-5b"], default="cogvideox-2b", help="CogVideoX model")
    parser.add_argument('--compile', action='store_true', help='Compile the model')
    parser.add_argument('--attention_type', type=str, default='sdpa', choices=['sdpa', 'sage', 'fa3', 'fa3_fp8'], help='Attention type')
    parser.add_argument("--start", type=int, default=0, help="Starting prompt id of this run.")
    parser.add_argument("--end", type=int, default=12, help="Ending prompt id of this run.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    if args.model == "cogvideox-2b":
        model_path = "THUDM/CogVideoX-2b"
        num_frames = 49
        torch_dtype = torch.float16
    else:
        model_path = "THUDM/CogVideoX1.5-5B"
        num_frames = 81
        torch_dtype = torch.bfloat16

    if args.attention_type == 'sage':
        F.scaled_dot_product_attention = sageattn
    elif args.attention_type == 'fa3':
        from sageattention.fa3_wrapper import fa3
        F.scaled_dot_product_attention = fa3
    elif args.attention_type == 'fa3_fp8':
        from sageattention.fa3_wrapper import fa3_fp8
        F.scaled_dot_product_attention = fa3_fp8

    video_dir = f"videos/{args.model}/{args.attention_type}"
    os.makedirs(video_dir, exist_ok=True)

    with open(prompt_path, "r", encoding="utf-8") as file:
        prompts = file.readlines()
    selected_prompts = [p.strip() for p in prompts[args.start:args.end]]

    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=torch_dtype)

    if args.compile:
        pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    for local_i, prompt in tqdm(enumerate(selected_prompts), total=len(selected_prompts)):
        global_i = args.start + local_i
        video = pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=50,
            num_frames=num_frames,
            guidance_scale=6,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).frames[0]

        export_to_video(video, f"{video_dir}/{global_i}.mp4", fps=8)
        del video
        gc.collect()
        torch.cuda.empty_cache()
