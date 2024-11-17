import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from sageattention import sageattn, sageattn_qk_int8_pv_fp16_cuda, sageattn_qk_int8_pv_fp8_cuda
import torch.nn.functional as F

F.scaled_dot_product_attention = sageattn_qk_int8_pv_fp8_cuda

prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."

pipe = CogVideoXPipeline.from_pretrained(
    "/root/autodl-tmp/CogVideoX-2b",
    torch_dtype=torch.float16
).to("cuda")

pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

torch.cuda.cudart().cudaProfilerStart()

video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

torch.cuda.cudart().cudaProfilerStop()

export_to_video(video, "output.mp4", fps=8)
