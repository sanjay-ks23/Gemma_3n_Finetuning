# Fine-tuning Gemma 3 4B with Unsloth

This guide provides a step-by-step approach to setting up and using a Conda environment with a GPU to fine-tune the Gemma 3 4B parameter model using Unsloth.

## 1. Environment Setup Summary

We have already completed the following steps:

1.  **Checked for GPU:** Verified the presence of an NVIDIA GeForce RTX 3070 with CUDA 12.9.
2.  **Created Conda Environment:** Created a Conda environment named `unsloth_env` with Python 3.10.
3.  **Installed Dependencies:** Installed all the necessary libraries, including `unsloth`, `torch`, `transformers`, and others, inside the `unsloth_env` environment.
4.  **Generated `requirements.txt`:** Created a `requirements.txt` file listing all the installed packages for reproducibility.

## 2. Activating the Conda Environment

To start using the environment, you need to activate it. Open your terminal and run the following command:

```bash
conda activate unsloth_env
```

Your terminal prompt should now be prefixed with `(unsloth_env)`, indicating that the environment is active.

## 3. Verifying the GPU Setup

To ensure that PyTorch can correctly access and use your GPU, you can run the following Python script. Create a file named `verify_gpu.py` and add the following code:

```python
import torch

if torch.cuda.is_available():
    print("GPU is available!")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available.")
```

Now, run the script from your terminal:

```bash
python verify_gpu.py
```

If the setup is correct, you should see an output indicating that the GPU is available, along with its details.

## 4. Deactivating the Conda Environment

Once you have finished your work, you can deactivate the Conda environment by running:

```bash
conda deactivate
```

This will return you to your system's default shell.

## 5. `requirements.txt`

For reference, here is the content of the `requirements.txt` file, which you can use to reinstall the exact same environment in the future:

```
accelerate==0.33.0
aiohappyeyeballs==2.6.1
aiohttp==3.12.14
aiosignal==1.4.0
async-timeout==5.0.1
attrs==25.3.0
bitsandbytes==0.43.2
certifi==2025.7.14
charset-normalizer==3.4.2
datasets==2.20.0
dill==0.3.8
docstring_parser==0.17.0
filelock==3.18.0
frozenlist==1.7.0
fsspec==2024.5.0
hf-xet==1.1.5
huggingface-hub==0.33.4
idna==3.10
Jinja2==3.1.6
markdown-it-py==3.0.0
MarkupSafe==3.0.2
mdurl==0.1.2
mpmath==1.3.0
multidict==6.6.3
multiprocess==0.70.16
networkx==3.4.2
numpy==1.26.4
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvrtc-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.1.105
nvidia-cudnn-cu12==8.9.2.26
nvidia-cufft-cu12==11.0.2.54
nvidia-curand-cu12==10.3.2.106
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse-cu12==12.1.0.106
nvidia-nccl-cu12==2.20.5
nvidia-nvjitlink-cu12==12.9.86
nvidia-nvtx-cu12==12.1.105
packaging==25.0
pandas==2.3.1
peft==0.12.0
propcache==0.3.2
psutil==7.0.0
pyarrow==21.0.0
pyarrow-hotfix==0.7
Pygments==2.19.2
python-dateutil==2.9.0.post0
pytz==2025.2
PyYAML==6.0.2
regex==2024.11.6
requests==2.32.4
rich==14.0.0
safetensors==0.5.3
shtab==1.7.2
six==1.17.0
sympy==1.14.0
tokenizers==0.20.3
torch==2.3.1
tqdm==4.67.1
transformers==4.45.0
triton==2.3.1
trl==0.9.6
typeguard==4.4.4
typing_extensions==4.14.1
tyro==0.9.26
tzdata==2025.2
unsloth @ git+https://github.com/unslothai/unsloth.git@3547416e16aec674e253ccba25cf0b1d9c2896c2
urllib3==2.5.0
xxhash==3.5.0
yarl==1.20.1
```