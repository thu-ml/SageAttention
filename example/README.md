
## **Plug-and-play Example**

**We can replace `scaled_dot_product_attention` easily.**  
We will take [CogvideoX](https://huggingface.co/THUDM/CogVideoX-2b) as an example:

**Just add the following codes and run!**
```python
from sageattention import sageattn
import torch.nn.functional as F

F.scaled_dot_product_attention = sageattn
```

Specifically,

```bash
cd example
python sageattn_cogvideo.py --compile
```

**You can get a lossless video in** `./example` **faster than by using** `python original_cogvideo.py --compile`.

---

### Another Example for cogvideoX-2B SAT
We will take [Cogvideo SAT](https://github.com/THUDM/CogVideo/tree/main) as an example:

Once you have set up the environment for cogvideoX's SAT and can generate videos, you can plug SageAttention and play easily by replacing lines 67-72 in CogVideo/sat/sat/transformer_defaults.py:


```python
67 |      attn_output = torch.nn.functional.scaled_dot_product_attention(
68 |          query_layer, key_layer, value_layer, 
69 |          attn_mask=None,
70 |          dropout_p=dropout_p,
71 |          is_causal=not is_full
72 |      )
```

with the following code:

```python
    attn_output = sageattn(
        query_layer, key_layer, value_layer, 
        is_causal=not is_full
    )
```
## Parallel SageAttention Inference

Install xDiT(xfuser >= 0.3.5) and diffusers(>=0.32.0.dev0) from sources and run:

```bash
# install latest xDiT(xfuser).
pip install "xfuser[flash_attn]"
# install latest diffusers (>=0.32.0.dev0), need by latest xDiT.
git clone https://github.com/huggingface/diffusers.git
cd diffusers && python3 setup.py bdist_wheel && cd dist && python3 -m pip install *.whl
# then run parallel sage attention inference.
./run_parallel.sh
```


