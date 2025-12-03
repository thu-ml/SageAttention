import torch
import torch.nn.functional as F
import os
from diffusers import CogVideoXTransformer3DModel, CogVideoXPipeline
from sageattention import sageattn
import argparse
from diffusers.utils import export_to_video

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default="THUDM/CogVideoX1.5-5B", help='Model path')
parser.add_argument('--compile', action='store_true', help='Compile the model')
parser.add_argument('--attention_type', type=str, default='sdpa', choices=['sdpa', 'sage', 'fa3', 'fa3_fp8'], help='Attention type')
args = parser.parse_args()

if args.attention_type == 'sage':
    F.scaled_dot_product_attention = sageattn
elif args.attention_type == 'fa3':
    from sageattention.fa3_wrapper import fa3
    F.scaled_dot_product_attention = fa3
elif args.attention_type == 'fa3_fp8':
    from sageattention.fa3_wrapper import fa3_fp8
    F.scaled_dot_product_attention = fa3_fp8

width = 1360//8
height = 768//8
model_id = args.model_path

transformer = CogVideoXTransformer3DModel.from_pretrained(model_id, subfolder="transformer",
    sample_height= height,
    sample_width= width,
    torch_dtype=torch.bfloat16)

pipe = CogVideoXPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)

if args.compile:
    pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

prompt = "A serene night scene in a forested area. The first frame shows a tranquil lake reflecting the star-filled sky above. The second frame reveals a beautiful sunset, casting a warm glow over the landscape. The third frame showcases the night sky, filled with stars and a vibrant Milky Way galaxy. The video is a time-lapse, capturing the transition from day to night, with the lake and forest serving as a constant backdrop. The style of the video is naturalistic, emphasizing the beauty of the night sky and the peacefulness of the forest."

with torch.no_grad():
    frames = pipe(
        prompt, 
        num_videos_per_prompt=1,
        num_frames=81,
        guidance_scale=6.0,
        num_inference_steps=50,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).frames[0]

export_to_video(frames, f"cogvideox-1.5-5b_{args.attention_type}.mp4", fps=8)
