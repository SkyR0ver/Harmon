# Harmon: Harmonizing Visual Representations for Unified Multimodal Understanding and Generation

![](data/method.png)

> **[Harmonizing Visual Representations for Unified Multimodal Understanding and Generation](https://arxiv.org/abs/2503.21979)**
>
> Size Wu, Wenwei Zhang, Lumin Xu, Sheng Jin, Zhonghua Wu, Qingyi Tao, Wentao Liu, Wei Li, Chen Change Loy
>
> [![arXiv](https://img.shields.io/badge/arXiv-2503.21979-b31b1b.svg)](https://arxiv.org/abs/2503.21979)
> [![Project Page](https://img.shields.io/badge/Project-Page-green)](https://wusize.github.io/projects/Harmon)
> [![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-orange)](https://huggingface.co/wusize/Harmon-1_5B)
> [![HuggingFace Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/wusize/Harmon)
> [![Bibtex](https://img.shields.io/badge/Cite-BibTeX-blue)](https://github.com/wusize/Harmon?tab=readme-ov-file#-citation)

## Introduction

**Harmon** is a novel unified framework for multimodal understanding and generation. Unlike existing state-of-the-art
architectures that disentangle visual understanding and generation with different encoder models, the proposed framework harmonizes
the visual presentations of understanding and generation via a shared MAR encoder. Harmon achieves advanced generation
performance on mainstream text-to-image generation benchmarks, and exhibits competitive results on multimodal understanding
tasks. In this repo, we provide inference code to run Harmon for image understanding (image-to-text) and text-to-image
generation, with two model variants Harmon-0.5B and Harmon-1.5B.

## 🚀 Project Status

| Task | Status |
|------|--------|
| 🛠️ Inference Code & Model Checkpoints | ✅ Released |
| 🌐 Project Page | ✅ Finished |
| 🤗 Online Demo |  ✅ [Finished](https://huggingface.co/spaces/wusize/Harmon) |
| 🔄 Finetuning Code | ✅ Released |


## Usage

### 📦 Required Packages
```text
mmengine
transformers==4.45.2
timm==0.9.12
flash_attn==2.3.4
```

### 📥 Checkpoints

Download the model checkpoints from 🤗 [wusize/harmon](https://huggingface.co/wusize/harmon) and organize them as follows:
```text
Harmon/
├── checkpoints
    ├── kl16.ckpt
    ├── harmon_0.5b.pth
    ├── harmon_1.5b.pth
```
It is recommended to use the following command to download the checkpoints
```bash
# pip install -U "huggingface_hub[cli]"
huggingface-cli download wusize/harmon  --local-dir checkpoints --repo-type model
```

### 🖌️ Image-to-text Generation

```shell
export PYTHONPATH=./:$PYTHONPATH
python scripts/image2text.py configs/models/qwen2_5_1_5b_kl16_mar_h.py \
         --checkpoint checkpoints/harmon_1.5b.pth  --image_size 512 \
         --image data/view.jpg --prompt "Describe the image in detail."
```

### 🖼️ Text-to-image Generation

You can generate images from text prompts using the following command:

```shell
export PYTHONPATH=./:$PYTHONPATH
python scripts/text2image.py configs/models/qwen2_5_1_5b_kl16_mar_h.py \
         --checkpoint checkpoints/harmon_1.5b.pth  --image_size 512 \
         --prompt 'a dog on the left and a cat on the right.'  --output output.jpg
```

To generate a list of images based on prompts in a json file.
```shell
export PYTHONPATH=./:$PYTHONPATH
accelerate launch scripts/batch_text2image.py configs/models/qwen2_5_1_5b_kl16_mar_h.py \
       --checkpoint checkpoints/harmon_1.5b.pth  --image_size 512 \
       --data path/to/xxx.json --output output --batch_size 4 --grid_size 2
```
The json file should look like:

```json
[
  {
   "prompt": "a dog on the left and a cat on the right."
  }
]
```


### 🤗 Loading Models from Huggingface

We have also converted our models to Huggingface format. You can directly load Harmon models from Huggingface using the `transformers` library:

```
from transformers import AutoTokenizer, AutoModel
harmon_tokenizer = AutoTokenizer.from_pretrained("wusize/Harmon-0_5B",
                                                 trust_remote_code=True)
harmon_model = AutoModel.from_pretrained("wusize/Harmon-0_5B",
                                         trust_remote_code=True).eval().cuda().bfloat16()
```

For more information on the usage of HF-based models, refer to the model cards in 

| Model Variant | LLM | MAR | Hugging Face Hub |
|:-------------:|:---:|:---:|:----------------:|
| **Harmon-0.5B** | Qwen2.5-0.5B-Instruct | MAR-Base | [![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-orange)](https://huggingface.co/wusize/Harmon-0_5B) |
| **Harmon-1.5B** | Qwen2.5-1.5B-Instruct | MAR-Huge | [![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-orange)](https://huggingface.co/wusize/Harmon-1_5B) |


### 🔄 Finetuning Harmon

For instructions on how to finetune Harmon models on your custom datasets, please refer to our detailed guide in [FINETUNE.md](FINETUNE.md).


## 📚 Citation

If you find Harmon useful for your research or applications, please cite our paper using the following BibTeX:

```bibtex
@misc{wu2025harmon,
      title={Harmonizing Visual Representations for Unified Multimodal Understanding and Generation}, 
      author={Size Wu and Wenwei Zhang and Lumin Xu and Sheng Jin and Zhonghua Wu and Qingyi Tao and Wentao Liu and Wei Li and Chen Change Loy},
      year={2025},
      eprint={2503.21979},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.21979}, 
}
```

## 📜 License
This project is licensed under [NTU S-Lab License 1.0](LICENSE).


## 🙏 Acknowledgement
The project builds upon the following open-source efforts:
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5): We use LLMs from Qwen2.5, including Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct.

- [MAR](https://github.com/LTH14/mar): The image generation pipeline is retrofitted from MAR.
