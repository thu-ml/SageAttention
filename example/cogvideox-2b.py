import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from sageattention import sageattn
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default="THUDM/CogVideoX-2b", help='Model path')
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

prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."

pipe = CogVideoXPipeline.from_pretrained(
    args.model_path,
    torch_dtype=torch.float16
).to("cuda")

if args.compile:
    pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

with torch.no_grad():
    video = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=49,
        guidance_scale=6,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).frames[0]

export_to_video(video, f"cogvideox-2b_{args.attention_type}.mp4", fps=8)
