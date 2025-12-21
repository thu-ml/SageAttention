
## Plug-and-play Example

This folder contains **plug-and-play inference scripts** for five Diffusers video models replacing full attention with **sageattn**.

Supported models:

* **CogVideoX:** [CogVideoX-2B](https://huggingface.co/zai-org/CogVideoX-2b) and [CogVideoX1.5-5B](https://huggingface.co/zai-org/CogVideoX1.5-5B)
* **WAN:** [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers), [Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers) and [Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers)
* [**HunyuanVideo**](https://huggingface.co/hunyuanvideo-community/HunyuanVideo)
* [**Mochi**](https://huggingface.co/genmo/mochi-1-preview)
* **LTX-Video:** [0.9.7-dev](https://huggingface.co/Lightricks/LTX-Video-0.9.7-dev) and [spatial upscaler](https://huggingface.co/Lightricks/ltxv-spatial-upscaler-0.9.7)

**We can replace `scaled_dot_product_attention` easily.**  
We will take [CogvideoX](ttps://huggingface.co/zai-org/CogVideoX-2b) as an example:

**Just add the following codes and run!**
```python
from sageattention import sageattn
import torch.nn.functional as F

F.scaled_dot_product_attention = sageattn
```

Specifically,

```bash
cd example
python cogvideox_infer.py --model cogvideox-2b --compile --attention_type sage
```

**You can get a lossless video in** `./example/videos/<model>/<attention_type>/` **faster than by using** `--attention_type sdpa.`.

> **Note:** If you set `--compile`, the first run will be slower than the following runs. Please run it twice to get the accurate speed.

> **Note:** `torch.compile` is generally incompatible with `enable_sequential_cpu_offload()`. Don't use them together.

## Modify Attention From Source Code
To have finer control over where to use SageAttention, you can modify a small subset of the source code. For example, in [`modify_mochi.py`](./modify_model/modify_mochi.py), you can replace the `MochiAttnProcessor2_0` from diffusers with your own attention class.

![Local Image](../assets/mochi_example.png)

![Local Image](../assets/hunyuanvideo_example.png)

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


