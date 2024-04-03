# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

from pathlib import Path
from typing import Literal
from configs.methods.common import Common
from vlpers.methods.sdm import SDM

class Method(SDM, Common):
    device = 'cuda'
    concepts_description:Literal['coarse', 'discriminative', 'rich'] = 'discriminative'

    pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"
    inference_steps=100
    guidance=7.5
    negative_prompt = [""]
    samples_per_label = 4

    concepts_path = ''
    cached_img_path = None