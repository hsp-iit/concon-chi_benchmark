# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

from typing import Literal
from configs.common import CommonConfig, project_root
from methods.clip_baselines import ImageOnly


class Config(CommonConfig):
    
    
    class Method(ImageOnly):
        device = 'cuda'
        save_concepts_to = None
        concepts_path = None # project_root / 'checkpoints/baselines/train1/parameters.npy'