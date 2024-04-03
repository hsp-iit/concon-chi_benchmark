# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

from configs.common import CommonConfig, project_root
from datasets.conconchi import rich_concepts
from methods.searle_sdm import SearleSDM

class Config(CommonConfig):

    class Method(SearleSDM):
        device = 'cuda'
        concepts_names = rich_concepts

        pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"
        inference_steps=100
        guidance=50
        negative_prompt = [""]
        samples_per_label = 4

        concepts_path = project_root / 'checkpoints/palavra'
        cached_img_path = None