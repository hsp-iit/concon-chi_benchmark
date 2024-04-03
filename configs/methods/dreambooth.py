# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

from pathlib import Path
from vlpers.methods.dreambooth import DreamBooth
from clearconf import BaseConfig
from clearconf import Hidden

class Method(DreamBooth, BaseConfig):
    # Diffusers Parameters
    class Diffusers:
        pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" # will take a while to download
        with_prior_preservation=True
        
        prior_loss_weight=1.0
        resolution=512
        train_batch_size=1
        gradient_accumulation_steps=1
        learning_rate=1e-6
        lr_scheduler="constant"
        lr_warmup_steps=0
        num_class_images=200
        train_text_encoder=True
        center_crop=True
        enable_xformers_memory_efficient_attention=True
        use_8bit_adam=True

    # Our Parameters
    inference_steps=100
    guidance=7.5
    negative_prompt = [""]
    samples_per_label = 4
    train_steps:Hidden = [400, 400, 600, 850, 450, 250, 700, 250, 500, 600, 450, 500, 1050, 300, 600, 450, 450, 450, 300, 650]

    class_data_dir= '[eval]cfg.Logging.exp_dir / "prior_preservation"'
    concepts_path = None
    cached_img_path = None