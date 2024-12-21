import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
import argparse

parser = argparse.ArgumentParser(description='CogVideoX example')
parser.add_argument('--compile', action='store_true', help='Compile the model')
args = parser.parse_args()

prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-2b",
    torch_dtype=torch.float16
).to("cuda")

if args.compile:
    pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

export_to_video(video, "cogvideo_sdpa.mp4", fps=8)
