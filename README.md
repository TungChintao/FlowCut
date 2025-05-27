<div align="center">
  <h1 style="display: inline-block; margin: 0;"><img src="images/icon.png" width="62" height="38" style="margin-top: -3px;">FlowCut: Rethinking Redundancy via Information Flow for Efficient Vision-Language Models</h1>
</div>


<h4 align="center"> 
Jintao Tong<sup>1</sup>,
Wenwei Jin<sup>2</sup>, 
Penda Qin<sup>2</sup>, 
Anqi Li<sup>3</sup>, 
Yixiong Zou<sup>1✉</sup>, <br>
Yuhong Li<sup>2✉</sup>,
Yuhua Li<sup>1</sup>,
Ruixuan Li<sup>1</sup>,

<sup>1</sup>School of Computer Science and Technology, Huazhong University of Science and Technology, <br><sup>2</sup>Xiaohongshu Inc., <sup>3</sup>Institute of Information Science, Beijing Jiaotong University

</h4>

<div align="center">
[![arXiv](https://img.shields.io/badge/Arxiv-2505.19536-AD1C18.svg?logo=arXiv)](https://arxiv.org/pdf/2505.19536)
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-yellow.svg)](https://github.com/TungChintao/FlowCut/blob/main/LICENSE)
</div>

## 🔥 News

*   Checkpoints for LLaVA-1.5-7B-FlowCut (retain 128 tokens/192 tokens) will be released soon!
*   [Code](https://github.com/TungChintao/FlowCut) will be released soon!
* **`2025.05.27`** We release our latest work [FlowCut](https://arxiv.org/abs/2505.19536), a plug-and-play, training-free token reduction method that seamlessly integrates into various VLMs for efficient training and inference.

## 💡 Highlights
<p align='center'>
<img src='images/intro.png' alt='mask' width='950px'>
</p>


> **TLDR:** To address inefficiency from excessive visual tokens in LVLMs, we propose a unified, bottom-up perspective based on information-flow, revealing dynamic redundancy emergence and introduce FlowCut, making pruning decision aligned with the model's inherent behavior, outperforming all existing approaches.

## 🛠 Preparation

Our code is easy to use.

1. Clone the [LLaVA](https://github.com/haotian-liu/LLaVA)'s repository.

```
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
```

2. Install the [LLaVA](https://github.com/haotian-liu/LLaVA)'s environment.

```
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  
pip install -e .
pip install flash-attn --no-build-isolation
```

3. For formal usage, you can install the package from PyPI by running the following command:

```
pip install flowcut
```

For development, you can install the package by cloning the repository and running the following command:

```
git clone https://github.com/TungChintao/flowcut
cd flowcut
pip install -e .
```

File organization as follow:

```
├── LLaVA-main
    ├── flowcut
    ├── llava
    ├── playground
		├── script
```

## 🚀 Quick Start

```Python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from FlowCut import flowcut
model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)
## FlowCut retains 64 visual tokens
model = flowcut(model, target_num=64)
```

## 📖 Evaluation

The evaluation code follows the structure of [LLaVA](https://github.com/haotian-liu/LLaVA) or [Lmms-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). After loading the model, simply add two lines as shown below:

```python
## Load LLaVA Model (code from llava.eval.model_vqa_loader)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
## add FlowCut
from flowcut import flowcut
model = flowcut(model, target_num=64)
```

Script templetes (please follow the detailed instruction in [LLaVA-Evaluation](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md)).

```Shell
bash scripts/v1_5/eval/[Benchmark].sh
```

Examples:

```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mme.sh
```

```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/pope.sh
```

```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/textvqa.sh
```

## 🎯 Training

The training code follows the structure of [LLaVA](https://github.com/haotian-liu/LLaVA). After loading the model, simply add two lines as shown below:

```python
## Load LLaVA Model (code from llava.train)
code of loading model...
## add FlowCut
from flowcut import flowcut
model = flowcut(model, target_num=64)
## training
trainer = LLaVATrainer(model=model,
                tokenizer=tokenizer,
                args=training_args,
                **data_module)
```

## 🔑 License

- This project is released under the [Apache 2.0 license](https://github.com/TungChintao/FlowCut/blob/main/LICENSE).

## 📌 Citation

- If you find this project useful in your research, please consider citing:

```bibtex
@article{tong2025flowcut,
  title={FlowCut: Rethinking Redundancy via Information Flow for Efficient Vision-Language Models}, 
  author={Jintao Tong and Wenwei Jin and Pengda Qin and Anqi Li and Yixiong Zou and Yuhong Li and Yuhua Li and Ruixuan Li},
  journal={arXiv preprint arXiv:2505.19536},
  year={2025}
}
```


## 👍 Acknowledgment
- This work is built upon [LLaVA](https://llava-vl.github.io/), [Qwen VL](https://github.com/QwenLM/Qwen2.5-VL), and [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA). We thank them for their excellent open-source contributions.

- We also thank [FastV](https://github.com/pkunlp-icler/FastV), [SparseVLM](https://github.com/Gumpest/SparseVLMs), [VisionZip](https://github.com/dvlab-research/VisionZip) and others for their contributions, which have provided valuable insights.
