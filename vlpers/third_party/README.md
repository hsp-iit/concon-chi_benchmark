The retrieval and generation methods that are automatically installed alongside the main package are detailed below.
The following table contains links to the original repositories and instructions on how to obtain a checkpoint for each model.

| Method | Source | Additional Notes |
| --- | --- | --- |
| [Pic2Word](https://github.com/google-research/composed_image_retrieval) | Download [here](https://drive.google.com/file/d/1IxRi2Cj81RxMu0ViT4q4nkfyjbSHm1dF/view?usp=sharing) | |
| [PALAVRA](https://github.com/NVlabs/PALAVRA) | Follow [this](https://github.com/NVlabs/PALAVRA#train-f_theta) instructions to generate a checkpoint | When PALAVRA is installed with our `setup.py`, the applied patch uses ViT-L/14 by default |
| [SEARLE](https://github.com/miccunifi/SEARLE) | Automatically downloaded | |
| [CLIP Baselines](https://github.com/openai/CLIP) | Automatically downloaded | |
| [Dreambooth](https://github.com/huggingface/diffusers) | Automatically downloaded | |
| [Textual Inversion](https://github.com/rinongal/textual_inversion) | LDM: put [this](https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt) in `checkpoints/textual-inversion/ldm.ckpt`<br> SDM: put [this](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/blob/main/sd-v1-4.ckpt) in `checkpoints/textual-inversion/smd.ckpt` | Textual Inversion uses the [ldm environment](https://github.com/rinongal/textual_inversion/blob/main/environment.yaml) that must be installed alongside with the vlpers environment. The installation path must coincide with the variable `ldm_env_interpreter_path` in the Textual Inversion configuration. For ```ldm``` at least 18GB of RAM are needed, while for ```sdm``` at least 24 GB of VRAM are needed. Set ```print_debug``` to True in the textual_inversion config to debug this method. |


> [!NOTE]  
> Some methods are cloned, checked out at a specific commit and then patched to make them installable.
> You can find the python script that performs the installation [here](install.py).
