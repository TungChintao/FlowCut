<div align="center">
  <h1 style="display: inline-block; margin: 0;">
    <img src="images/icon.png" width="58" height="38" align="absmiddle">FlowCut: Rethinking Redundancy via Information Flow for Efficient Vision-Language Models
  </h1>
</div>


<h4 align="center"> 
Jintao Tong<sup>1</sup>,
Wenwei Jin<sup>2</sup>, 
Pengda Qin<sup>2</sup>, 
Anqi Li<sup>3</sup>, 
Yixiong Zou<sup>1✉</sup>,<br>
Yuhong Li<sup>2✉</sup>,
Yuhua Li<sup>1</sup>,
Ruixuan Li<sup>1</sup>
<br><br> 
<sup>1</sup>School of Computer Science and Technology, Huazhong University of Science and Technology<br> <sup>2</sup>Xiaohongshu Inc., <sup>3</sup>Institute of Information Science, Beijing Jiaotong University

</h4>

<div align="center">
	
[![arXiv](https://img.shields.io/badge/arXiv-2505.19536-AD1C18.svg?logo=arXiv)](https://arxiv.org/pdf/2505.19536)
[![HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoints-orange)](https://huggingface.co/JosephTong/llava-v1.5-7b-flowcut192)
[![License](https://img.shields.io/badge/📃%20License-Apache_2.0-yellow.svg)](https://github.com/TungChintao/FlowCut/blob/main/LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/TungChintao/FlowCut?style=social)](https://github.com/TungChintao/FlowCut/stargazers)

</div>

## 🔥 News

* **`2025.07.01`** 🚀 We release the implementation of FlowCut for Qwen2-VL! See details [here](https://github.com/TungChintao/FlowCut/tree/main/FlowCut_Qwen).
* **`2025.05.29`** 🤗 The checkpoints of [llava-v1.5-7b-flowcut128](https://huggingface.co/JosephTong/llava-v1.5-7b-flowcut128) and [llava-v1.5-7b-flowcut192](https://huggingface.co/JosephTong/llava-v1.5-7b-flowcut192), retaining 128 and 192 visual tokens respectively, have been released!
* **`2025.05.28`** 🚀 [Code](https://github.com/TungChintao/FlowCut) is available, and FlowCut can be easily installed with `pip install flowcut`！
* **`2025.05.26`** 📝 We release our latest work [FlowCut](https://arxiv.org/abs/2505.19536), a plug-and-play, training-free token reduction method that seamlessly integrates into various VLMs for efficient training and inference.

## 💡 Highlights
<p align='center'>
<img src='https://github.com/TungChintao/FlowCut/blob/main/images/intro.png' alt='mask' width='950px'>
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
git clone https://github.com/TungChintao/FlowCut
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
from flowcut import flowcut
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
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval/vqav2.sh
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
  author={Tong, Jintao and Jin, Wenwei and Qin, Pengda and Li, Anqi and Zou, Yixiong and Li, Yuhong and Li, Yuhua and Li, Ruixuan},
  journal={arXiv preprint arXiv:2505.19536},
  year={2025}
}
```


## 👍 Acknowledgment
- This work is built upon [LLaVA](https://llava-vl.github.io/), [Qwen VL](https://github.com/QwenLM/Qwen2.5-VL), and [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA). We thank them for their excellent open-source contributions.

- We also thank [FastV](https://github.com/pkunlp-icler/FastV), [SparseVLM](https://github.com/Gumpest/SparseVLMs), [VisionZip](https://github.com/dvlab-research/VisionZip) and others for their contributions, which have provided valuable insights.
