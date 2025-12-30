import torch, os, gc
import argparse
from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_image, load_video
from tqdm import tqdm
from modify_model.modify_ltx import set_sage_attn_ltx
from diffusers.models.attention_dispatch import dispatch_attention_fn
from sageattention import sageattn


ATTN = {
    "sdpa": dispatch_attention_fn,
    "sage": sageattn,
}
prompt_path = "videos/testing_prompts.txt"


def round_to_nearest_resolution_acceptable_by_vae(height, width, pipe):
    height = height - (height % pipe.vae_spatial_compression_ratio)
    width = width - (width % pipe.vae_spatial_compression_ratio)
    return height, width


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compile', action='store_true', help='Compile the model')
    parser.add_argument('--attention_type', type=str, default='sdpa', choices=['sdpa', 'sage'], help='Attention type')
    parser.add_argument("--start", type=int, default=0, help="Starting prompt id of this run.")
    parser.add_argument("--end", type=int, default=12, help="Ending prompt id of this run.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    video_dir = f"videos/ltx/{args.attention_type}"
    os.makedirs(video_dir, exist_ok=True)

    with open(prompt_path, "r", encoding="utf-8") as file:
        prompts = file.readlines()
    selected_prompts = [p.strip() for p in prompts[args.start:args.end]]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=device).manual_seed(42)

    pipe_id = "Lightricks/LTX-Video-0.9.7-dev"
    upscaler_id = "Lightricks/ltxv-spatial-upscaler-0.9.7"

    pipe = LTXConditionPipeline.from_pretrained(
        pipe_id,
        torch_dtype=torch.bfloat16,
    )
    set_sage_attn_ltx(pipe.transformer, ATTN[args.attention_type])

    # if args.compile:
    #     pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(
        upscaler_id,
        vae=pipe.vae,
        torch_dtype=torch.bfloat16,
    )
    pipe_upsample.enable_model_cpu_offload()

    for local_i, prompt in tqdm(enumerate(selected_prompts), total=len(selected_prompts)):
        global_i = args.start + local_i
        image = load_image(f"videos/ltx_first_frames/{global_i}.png")
        video = load_video(export_to_video([image]))  # compress the image using video compression as the model was trained on videos
        condition1 = LTXVideoCondition(video=video, frame_index=0)

        negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
        expected_height, expected_width = 480, 832
        downscale_factor = 2 / 3
        num_frames = 96

        # Part 1. Generate video at smaller resolution
        downscaled_height, downscaled_width = int(expected_height * downscale_factor), int(expected_width * downscale_factor)
        downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(downscaled_height, downscaled_width, pipe)
        latents = pipe(
            conditions=[condition1],
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=downscaled_width,
            height=downscaled_height,
            num_frames=num_frames,
            num_inference_steps=30,
            generator=gen,
            output_type="latent",  # Crucial: ask the pipeline to return latents, not decoded frames
        ).frames

        # Part 2. Upscale generated video using latent upsampler with fewer inference steps
        # The available latent upsampler upscales the height/width by 2x
        upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
        upscaled_latents = pipe_upsample(
            latents=latents,
            output_type="latent"
        ).frames

        # Part 3. Denoise the upscaled video with few steps to improve texture (optional, but recommended)
        video = pipe(
            conditions=[condition1],
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=upscaled_width,
            height=upscaled_height,
            num_frames=num_frames,
            denoise_strength=0.4,  # Effectively, 4 inference steps out of 10
            num_inference_steps=10,
            latents=upscaled_latents,
            decode_timestep=0.05,
            image_cond_noise_scale=0.025,
            generator=gen,
            output_type="pil",
        ).frames[0]

        # Part 4. Downscale the video to the expected resolution
        video = [frame.resize((expected_width, expected_height)) for frame in video]
        export_to_video(video, f"{video_dir}/{global_i}.mp4", fps=24)
        del video, latents, upscaled_latents, condition1
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
