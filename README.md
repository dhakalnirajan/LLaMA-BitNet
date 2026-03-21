# <span style="color:#fa362f">Welcome to LLaMA-BitNet</span>

<span style="color:#ffa042">Welcome to the LLaMA-BitNet repository. Our repository is your gateway to training your very own BitNet model, as highlighted in the groundbreaking paper [<span style="color:#52d1ff;text-decoration: underline;">The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764</span>). Built upon the cutting-edge [<span style="color:#52d1ff;text-decoration: underline;">LLaMA 2](https://llama.meta.com</span>) architecture, this project allows you to unleash the potential of a model wielding approximately 78 million parameters, trained on a staggering corpus of around 1.5 billion tokens.</span>

<br>

> Note: You need to have access to LLaMA model if you wish to run code without modifications. To get access to LLaMA family of models, you need to go to [https://llama.meta.com/llama-downloads/](https://llama.meta.com/llama-downloads/) and provide credentials which you use in Hugging Face. After that, you will receive mail to either download weights directly to your device or to use LLaMA through API.

<br>

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-transformers%20%7C%20datasets%20%7C%20Models-blueviolet?style=for-the-badge&logo=huggingface&labelColor=ee4c2c&color=ff9d00)](https://huggingface.co/)
![GitHub License](https://img.shields.io/github/license/dhakalnirajan/LLaMA-BitNet?style=for-the-badge&logo=github&logoColor=white&label=License&labelColor=purple&color=orange)
[![Repository Stars](https://img.shields.io/github/stars/dhakalnirajan/LLaMA-BitNet)](https://github.com/dhakalnirajan/LLaMA-BitNet/stargazers)
![GitHub forks](https://img.shields.io/github/forks/dhakalnirajan/LLaMA-BitNet)
[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm.svg)](https://huggingface.co/datasets)
[![Follow me on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-me-on-HF-sm.svg)](https://huggingface.co/nirajandhakal)[![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Ftwitter.com%2Fnirajandhakal_7&style=for-the-badge&logo=X&logoColor=blue&label=Follow%20%40nirajandhakal_7%20&labelColor=black&color=black&link=https%3A%2F%2Ftwitter.com%2Fnirajandhakal_7)](https://twitter.com/nirajandhakal_7)
<img src="https://chunte-hfba.static.hf.space/images/Outlined%20Huggies/Greeting%20Huggy%20left.png" alt="HuggingFace Logo" height=80 width=80>

## <span style="color:#ff52fc">Easy Installation</span>

Getting started with LLaMA-BitNet is a breeze! Follow these simple steps to install all the necessary modules:

```shell
pip install -r requirements.txt
```

## <span style="color:#5e6eff">Intuitive File Structure</span>

Our repository boasts a clear and intuitive file structure designed for effortless navigation and customization:

```
LLaMA-BitNet                    (root folder)
|
│   ├── inference.py            (Run inference with the trained BitNet model)
│   ├── LICENSE                 (MIT License)
│   ├── README.md
│   ├── requirements.txt        (List of required modules for installation)
│   ├── train.py                (Run the training process)
│   └── utils.py                (Contains utility functions)
```

## <span style="color:#00ff00">Training Data</span>

A 15% subset of the <span style="color:#ff006a">`OpenWebText2`</span> dataset meticulously prepared for training, this subset, tokenized with a context length of 256 for seamless testing, offers unparalleled versatility. However, our code also facilitates manual tokenization, allowing you to train on datasets of your choice effortlessly.

## <span style="color:#ff4592">Streamlined Dependencies</span>

We've curated a set of essential dependencies listed in the <span style="color:#ff006a">`requirements.txt`</span> file, ensuring a seamless installation process:

```text
transformers
datasets
torch
wandb
huggingface_hub
```

## Architecture
 
The conversion applied by `convert_to_bitnet()` makes exactly two changes to a standard LLaMA 2 decoder layer, as specified in the paper:
 
1. **Replace** every `nn.Linear` inside `self_attn` and `mlp` with `BitLinear`.
2. **Remove** `input_layernorm` and `post_attention_layernorm` from each decoder layer — because `BitLinear` now owns a parameter-free RMSNorm internally.
 
`embed_tokens` and `lm_head` stay full-precision. The rest of the LLaMA components (RoPE, SwiGLU, causal mask) are unchanged.
 
### BitLinear forward pass (training)
 
```
x  →  RMSNorm (parameter-free, inside BitLinear)
   →  per-token absmax quantise to int8 range [-127, 127]   (STE)
   →  F.linear(x_q, w_q)   where w_q = absmean-quantised to {-1, 0, +1}  (STE)
   →  rescale by (w_scale × act_scale)
```
 
The straight-through estimator (STE) lets gradients flow through the non-differentiable round() calls during backprop. High-precision shadow weights live in `self.weight` the whole time — only the quantised copies are used for the forward pass.
 
---



## Training
 
```bash
python train.py
```
 
Options:
 
```
--steps          N       Total training steps     (default: 10000)
--batch-size     N       Per-device batch size    (default: 32)
--lr             F       Learning rate            (default: 3e-4)
--checkpoint-dir PATH    Where to save models     (default: ./checkpoints)
--wandb                  Enable W&B logging
```
 
Checkpoints are saved to `./checkpoints/step_NNNNNNN/` every 1000 steps and a final model at `./checkpoints/final/`.
 
---
 
## Inference
 
```bash
# Interactive REPL
python inference.py --model ./checkpoints/final
 
# Single prompt
python inference.py --model ./checkpoints/final \
    --prompt "The future of computing is" \
    --max-new-tokens 200 \
    --temperature 0.8
```


## <span style="color:#ff45f9">Unleash the Full Potential of BitNet</span>

Our BitNet architecture is engineered for excellence, drawing inspiration from the meticulous design laid out in the training details manuscript, [<span style="color:#52d1ff;text-decoration: underline;">The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf](https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf)</span>. By seamlessly integrating BitLinear and leveraging HuggingFace's <span style="color:#ffff00; background-color:#112211;">`LlamaForCasualLM`</span>, we empower you to unlock the true power of BitNet.

Explore, train, and revolutionize with LLaMA-BitNet!
