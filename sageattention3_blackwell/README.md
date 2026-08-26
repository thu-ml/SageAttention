# SageAttention3
<!-- We are continuously updating more features. You could **Star** and **Watch** our repository to stay updated.

--- -->
This repository provides the official implementation of SageAttention3

**SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training**  
Paper: https://arxiv.org/abs/2505.11594  
Jintao Zhang, Jia Wei, Pengle Zhang, Xiaoming Xu, Haofeng Huang, Haoxu Wang, Kai Jiang, Jun Zhu, Jianfei Chen

# Limitaitions:
Currently, SageAttention3 works well for: 
1. Video generation models: CogVideoX-2B, HunyuanVideo, Mochi.
2. Almost all image generation models, including Flux and Stable-Diffusion3.5.

**Note: SageAttention3 does not guarantee lossless acceleration for all models. For other video generation models, we recommend selectively using SageAttention2++ in certain layers or timesteps.**  

For example:  
- Apply **SageAttention2++** only at the **first and last timesteps**,  
- Use **SageAttention3** for all the others.  

This hybrid approach may achieve **lossless acceleration**.  

## Installation
### Base environment
+ `python>=3.13`   , `torch>=2.8.0`, `CUDA >=12.8`

### Install Package

To use SageAttention3, please **compile from source**:
```
git clone https://github.com/thu-ml/SageAttention
cd SageAttention/sageattention3_blackwell 
python setup.py install
```


## How to Use
```python
from sageattn3 import sageattn3_blackwell
attn_output = sageattn3_blackwell(q, k, v, is_causal=False)
```
+ `q, k, v` are **FP16/BF16** dtype with the shape `(batch_size, head_num, seq_len, head_dim)` 
+ `is_causal` determines the use of a causal mask.

### Long non-causal sequences

With `per_block_mean=True`, SA3 creates a FP32 correction tensor named
`delta_s`. Its Blackwell TMA layout uses int32 dynamic strides. If the full
correction would overflow that stride, SA3 automatically evaluates aligned Q
chunks while reusing the quantized K and V tensors. This avoids the descriptor
failure reported in [issue #382](https://github.com/thu-ml/SageAttention/issues/382)
without changing the normal path for smaller inputs.

You can also request chunking explicitly to bound `delta_s` memory:

```python
attn_output = sageattn3_blackwell(
    q,
    k,
    v,
    is_causal=False,
    per_block_mean=True,
    q_chunk_size=1024,
)
```

`q_chunk_size` must be a positive multiple of 128. Chunking is limited to
non-causal, unmasked, per-block-mean attention because the current kernel API
does not expose an absolute query-position offset for causal chunks.

For `(B=1, H=56, N=73,774, D=128)`, a 1,024-token chunk reduces the analytical
FP32 `delta_s` allocation from about 8.89 GiB to 126.22 MiB.

## Performance
### Speed of Kernels
![Speed on RTX5090](../assets/sage3_speed.png)

### Video and Image Generation Examples
![Image Examples](../assets/sage3_result.png)



## Citation
**If you use this code or find our work valuable, please cite:**
```
@inproceedings{zhang2025sageattention,
  title={SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration}, 
  author={Zhang, Jintao and Wei, Jia and Zhang, Pengle and Zhu, Jun and Chen, Jianfei},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
@inproceedings{zhang2024sageattention2,
  title={Sageattention2: Efficient attention with thorough outlier smoothing and per-thread int4 quantization},
  author={Zhang, Jintao and Huang, Haofeng and Zhang, Pengle and Wei, Jia and Zhu, Jun and Chen, Jianfei},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
@article{zhang2025sageattention2++,
  title={Sageattention2++: A more efficient implementation of sageattention2},
  author={Zhang, Jintao and Xu, Xiaoming and Wei, Jia and Huang, Haofeng and Zhang, Pengle and Xiang, Chendong and Zhu, Jun and Chen, Jianfei},
  journal={arXiv preprint arXiv:2505.21136},
  year={2025}
}
@article{zhang2025sageattention3,
  title={SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training},
  author={Zhang, Jintao and Wei, Jia and Zhang, Pengle and Xu, Xiaoming and Huang, Haofeng and Wang, Haoxu and Jiang, Kai and Zhu, Jun and Chen, Jianfei},
  journal={arXiv preprint arXiv:2505.11594},
  year={2025}
}
```
