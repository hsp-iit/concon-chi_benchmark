# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

from typing import Literal

from configs.methods.common import Common
from vlpers.methods.clip_baselines import TextOnly

class Method(TextOnly, Common):
    concepts_description: Literal['coarse', 'discriminative', 'rich'] = 'discriminative'
    num_concepts = '[eval]cfg.Data.num_concepts'

    load_image_pool = None
    load_concepts = None
