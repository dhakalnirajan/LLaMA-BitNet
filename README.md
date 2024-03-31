# <span style="color:#fa362f">Welcome to LLaMA-BitNet</span>

<span style="color:#ffa042">Welcome to the LLaMA-BitNet repository, where you can dive into the fascinating world of BitNet models. Our repository is your gateway to training your very own BitNet model, as highlighted in the groundbreaking paper [<span style="color:#52d1ff;text-decoration: underline;">The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764</span>). Built upon the cutting-edge [<span style="color:#52d1ff;text-decoration: underline;">LLaMA 2](https://llama.meta.com</span>) architecture, this project allows you to unleash the potential of a model wielding approximately 78 million parameters, trained on a staggering corpus of around 1.5 billion tokens.</span>

<br>

> Note: You need to have access to LLaMA model if you wish to run code without modifications. To get access to LLaMA family of models, you need to go to [https://llama.meta.com/llama-downloads/](https://llama.meta.com/llama-downloads/) and provide credentials which you use in Hugging Face. After that, you will receive mail to either download weights directly to your device or to use LLaMA through API.

<br>

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-transformers%20%7C%20datasets%20%7C%20Models-blueviolet?style=for-the-badge&logo=huggingface)](https://huggingface.co/)
[![Repository Stars](https://img.shields.io/github/stars/dhakalnirajan/LLaMA-BitNet?style=for-the-badge)](https://github.com/dhakalnirajan/LLaMA-BitNet/stargazers)
[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm.svg)](https://huggingface.co/datasets)
[![Follow me on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-me-on-HF-sm.svg)](https://huggingface.co/nirajandhakal)
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

## <span style="color:#00ff00">Empowering Training Data</span>

Harness the power of a 15% subset of the <span style="color:#ff006a">`OpenWebText2`</span> dataset meticulously prepared for training. This subset, tokenized with a context length of 256 for seamless testing, offers unparalleled versatility. However, our code also facilitates manual tokenization, allowing you to train on datasets of your choice effortlessly.

## <span style="color:#ff4592">Streamlined Dependencies</span>

We've curated a set of essential dependencies listed in the <span style="color:#ff006a">`requirements.txt`</span> file, ensuring a seamless installation process:

```text
transformers
datasets
torch
wandb
huggingface_hub
```

## <span style="color:#ff45f9">Unleash the Full Potential of BitNet</span>

Our BitNet architecture is engineered for excellence, drawing inspiration from the meticulous design laid out in the training details manuscript, [<span style="color:#52d1ff;text-decoration: underline;">The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf](https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf)</span>. By seamlessly integrating BitLinear and leveraging HuggingFace's <span style="color:#ffff00; background-color:#112211;">`LlamaForCasualLM`</span>, we empower you to unlock the true power of BitNet.

Explore, train, and revolutionize with LLaMA-BitNet!
