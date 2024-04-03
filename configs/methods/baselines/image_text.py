# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

from configs.methods.common import Common
from vlpers.methods.clip_baselines import ImageText

class Method(ImageText, Common):
    device:str = "cuda"
    
    load_concepts = None
    load_image_pool = None
    num_concepts = '[eval]cfg.Data.num_concepts'
    
    alpha = 0.5
