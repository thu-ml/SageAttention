"""
modified from: https://github.com/xdit-project/xDiT/blob/main/examples/cogvideox_example.py
sh ./run_parallel.sh
"""

import torch
import torch.distributed
from xfuser import xFuserCogVideoXPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_runtime_state,
    is_dp_last_group,
)
from diffusers.utils import export_to_video
import time
import torch.nn.functional as F
from functools import partial
import sageattention

torch.set_grad_enabled(False)


def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    parser.add_argument("--use_sage_attn_fp16", action="store_true", help="Use Sage Attention fp16 or not.")
    parser.add_argument("--use_sage_attn_fp8", action="store_true", help="Use Sage Attention fp8 or not.")
    args = xFuserArgs.add_cli_args(parser).parse_args()

    engine_args = xFuserArgs.from_cli_args(args)
    # Check if ulysses_degree is valid
    num_heads = 30
    if engine_args.ulysses_degree > 0 and num_heads % engine_args.ulysses_degree != 0:
        raise ValueError(
            f"ulysses_degree ({engine_args.ulysses_degree}) must be a divisor of the number of heads ({num_heads})"
        )
    
    # Init distributed env here
    engine_config, input_config = engine_args.create_config()
    local_rank = get_world_group().local_rank
    
    sage_tag = "sage+None"
    if args.use_sage_attn_fp16:
        F.scaled_dot_product_attention = partial(
            sageattention.sageattn_qk_int8_pv_fp16_cuda, 
            pv_accum_dtype="fp32")
        sage_tag = f"sage+fp16"
    elif args.use_sage_attn_fp8:
        F.scaled_dot_product_attention = partial(
            sageattention.sageattn_qk_int8_pv_fp8_cuda, 
            pv_accum_dtype="fp32+fp32")
        sage_tag = f"sage+fp8" # acc fp32

    pipe = xFuserCogVideoXPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.bfloat16,
    )

    if args.enable_sequential_cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=local_rank)
    else:
        device = torch.device(f"cuda:{local_rank}")
        pipe = pipe.to(device)
    
    # Always enable tiling and slicing to avoid VAE OOM while batch size > 1
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    output = pipe(
        height=input_config.height,
        width=input_config.width,
        num_frames=input_config.num_frames,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        generator=torch.Generator().manual_seed(input_config.seed),
        guidance_scale=6,
        use_dynamic_cfg=True,
        latents=None # Load local latents or let it None.
    ).frames[0]

    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
        f"tp{engine_args.tensor_parallel_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}_"
        f"compile{engine_config.runtime_config.use_torch_compile}"
    )
    
    if is_dp_last_group():
        world_size = get_world_group().world_size
        prompt_tag: str = input_config.prompt[0]
        prompt_tag = prompt_tag.replace(" ", "_").replace(".", "")
        resolution = f"{input_config.width}x{input_config.height}x{input_config.num_frames}"
        output_filename = (f"results/cogvideox_{parallel_info}_{sage_tag}_{world_size}gpu_"
                           f"{resolution}_{prompt_tag}.mp4")
        export_to_video(output, output_filename, fps=8)
        print(f"output saved to {output_filename}")

    if get_world_group().rank == get_world_group().world_size - 1:
        print(f"epoch time: {elapsed_time:.2f} sec, memory: {peak_memory/1e9} GB")
    get_runtime_state().destory_distributed_env()


if __name__ == "__main__":
    main()
