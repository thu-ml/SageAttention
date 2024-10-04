# SageAttention

This repository provides the official implementation of SageAttention.

**SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration**  
Paper: https://arxiv.org/abs/2410.02367  
Jintao Zhang, Jia Wei, Pengle Zhang, Jun Zhu, Jianfei Chen


![Local Image](./resource/intro.png)

## Base environment
`python>=3.9`   
`torch>=2.3.0`  
`triton>=3.0.0` 

We recommend to install:

`python>=3.11`  
`torch>=2.4.0`  
`triton-nightly`


## Installation
Compile from source:
```
cd sageattention 
pip install .
```
Or you can install using pip:  (coming soon)  
```
pip install sageattention  
```




## How to use
```python
from sageattention import sageattn
attn_output = sageattn(q, k, v, is_causal=False, smooth_k=True)
```
`q, k, v` are FP16 data type with the shape `(batch_size, head_num, seq_len, head_dim)`. The parameter is_causal determines the use of a causal mask. `smooth_k` is a technique we proposed to ensure the accuracy. Disabling `smooth_k` might slightly increase speed, but could compromise accuracy if the distribution of `q, k, v` is irregular.

> **Note:** sageattn() is a accurate implementation that integrating smoothing K, INT8 per-block quantization for `q, k`, and a FP16 accumulator for Matmul of $PV$. 
Support for `head_dim` values of 64 and 128 is currently available. Extended support for values 48, 72, 96, and 256 will be available soon.





### Example
We will take [Cogvideo](https://github.com/THUDM/CogVideo/tree/main) as an example:

Once you have set up the environment for cogvideoX's SAT and can generate videos, you can plug SageAttention and play easily by replacing lines 66-73 in CogVideo/sat/sat/transformer_defaults.py:


```python
66 |  with context:
67 |      attn_output = torch.nn.functional.scaled_dot_product_attention(
68 |          query_layer, key_layer, value_layer, 
69 |          attn_mask=None,
70 |          dropout_p=dropout_p,
71 |          is_causal=not is_full
72 |      )
```

with the following code:

```python
from sageattention import sageattn
with context:
    attn_output = sageattn(
        query_layer, key_layer, value_layer, 
        is_causal=not is_full
    )
```


## Performance
### Speed of Kernels
![Local Image](./resource/4090_hd64.png)

![Local Image](./resource/4090_hd128.png)

![Local Image](./resource/3090_hd64.png)

![Local Image](./resource/3090_hd64.png)



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