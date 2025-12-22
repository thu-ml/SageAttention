# ================================================================
# SageAttention in HunyuanVideo
# ================================================================
# NOTE: This file is kept for reference and is currently DISABLED.
#
# Diffusers HunyuanVideo pipeline commonly uses `attention_mask`, which is not supported by `sageattn`.
# So a naive injection here will not work reliably (falls back to SDPA, or gives wrong results).
#
# What to do instead:
# (1) Official SageAttention implementation:
#     https://huggingface.co/tencent/HunyuanVideo-1.5
#     -> use CLI flags: `--use_sageattn` and `--sage_blocks_range`
#
# (2) Workaround / selective SageAttention (mask-free image tokens):
#     https://github.com/thu-ml/SageAttention/issues/115
#     -> split text vs image tokens; use Sage only on large image-token self-attention.
# ================================================================


# import torch, os, argparse, gc
# from tqdm import tqdm
# from diffusers import HunyuanVideoPipeline
# from diffusers.utils import export_to_video
# from sageattention import sageattn
# import torch.nn.functional as F
# from modify_model.modify_hunyuan import set_sage_attn_hunyuan

# ATTENTION = {
#     "sage": sageattn,
#     "sdpa": F.scaled_dot_product_attention,
# }

# os.environ["TOKENIZERS_PARALLELISM"]="false"
# prompt_path = "videos/testing_prompts.txt"

# def parse_args():
#     parser = argparse.ArgumentParser(description="Hunyuan Inference")
#     parser.add_argument("--model_path", default="hunyuanvideo-community/HunyuanVideo", help="Hunyuan model path")
#     parser.add_argument('--compile', action='store_true', help='Compile the model')
#     parser.add_argument('--attention_type', type=str, default='sdpa', choices=['sdpa', 'sage'], help='Attention type')
#     parser.add_argument("--start", type=int, default=0, help="Starting prompt id of this run.")
#     parser.add_argument("--end", type=int, default=12, help="Ending prompt id of this run.")
#     args = parser.parse_args()
#     return args

# if __name__ == "__main__":
#     args = parse_args()

#     video_dir = f"videos/hunyuan/{args.attention_type}"
#     os.makedirs(video_dir, exist_ok=True)

#     with open(prompt_path, "r", encoding="utf-8") as file:
#         prompts = file.readlines()
#     selected_prompts = [p.strip() for p in prompts[args.start:args.end]]

#     pipe = HunyuanVideoPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)

#     set_sage_attn_hunyuan(pipe.transformer, ATTENTION[args.attention_type])

#     # if args.compile:
#     #     pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

#     pipe.enable_sequential_cpu_offload()
#     pipe.vae.enable_slicing()
#     pipe.vae.enable_tiling()

#     for local_i, prompt in tqdm(enumerate(selected_prompts), total=len(selected_prompts)):
#         global_i = args.start + local_i
#         output = pipe(
#             prompt=prompt,
#             height=320,
#             width=512,
#             num_frames=61,
#             num_inference_steps=30,
#             generator=torch.Generator(device="cuda").manual_seed(42),
#         ).frames[0]
#         export_to_video(output, f"{video_dir}/{global_i}.mp4", fps=15)
#         del output
#         gc.collect()
#         torch.cuda.empty_cache()
