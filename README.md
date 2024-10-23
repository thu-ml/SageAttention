# SageAttention

This repository provides the official implementation of SageAttention.

**SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration**  
Paper: https://arxiv.org/abs/2410.02367  
Jintao Zhang, Jia Wei, Pengle Zhang, Jun Zhu, Jianfei Chen


![Local Image](./resource/intro.png)

## Base environment
`python>=3.9`   
`torch>=2.3.0`  
`triton>=2.3.0` 

We recommend to install: (the kernel will be faster a little)  
`python>=3.11`  
`torch>=2.4.0`  
`triton-nightly`


## Installation
Install using pip:  
```
pip install sageattention
```

Or compiling from source:
```
cd sageattention 
pip install .
```


> **Note:** SageAttention is currently optimized for RTX4090 and RTX3090 GPUs. Performance improvements may not be significant on other GPU architectures. We will progressively extend support to other GPUs.


## How to use
```python
from sageattention import sageattn
attn_output = sageattn(q, k, v, is_causal=False, smooth_k=True)
```
`q, k, v` are **FP16/BF16** type with the shape `(batch_size, head_num, seq_len, head_dim)`. `is_causal` determines the use of a causal mask. `smooth_k` is a technique we proposed to ensure the accuracy. Disabling `smooth_k` might slightly increase speed, but could compromise accuracy if the distribution of `q, k, v` is irregular.

> **Note:** sageattn() is an accurate implementation that integrating smoothing K, INT8 per-block quantization for `q, k`, and a FP16 accumulator for Matmul of $PV$. 
Support for `head_dim` values of `64`, `96`, and `128` is currently available. Extended support for values 48, 72, and 256 will be available soon.




## **Plug-and-play Example**

**We can replace `scaled_dot_product_attention` easily.**  
We will take [Cogvideo](https://huggingface.co/THUDM/CogVideoX-2b) as an example:

**Just add the following codes and run!**
```python
from sageattention import sageattn
import torch.nn.functional as F

F.scaled_dot_product_attention = sageattn
```

Specifically,

```bash
cd example
python sageattn_cogvideo.py
```

**You can get a lossless video in** `./example` **faster than by using** `python original_cogvideo.py`


## Performance
### Speed of Kernels
![Local Image](./resource/4090_hd64.png)

![Local Image](./resource/4090_hd128.png)

![Local Image](./resource/3090_hd64.png)

![Local Image](./resource/3090_hd64.png)

> **Note:** The TOPS results refer only to the Attention Kernel, excluding the quantization and smoothing K.

### End-to-end performance
![Local Image](./resource/real_speedup.png)

![Local Image](./resource/end-to-end_performance.png)


## Citation
If you use this code or find our work valuable, please cite:
```
@misc{zhang2024sageattentionaccurate8bitattention,
      title={SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration}, 
      author={Jintao Zhang and Jia wei and Pengle Zhang and Jun Zhu and Jianfei Chen},
      year={2024},
      eprint={2410.02367},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.02367}, 
}
```
