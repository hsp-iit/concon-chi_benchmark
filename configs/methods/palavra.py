# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

from typing import Literal
from vlpers.methods.palavra import Palavra
from pathlib import Path
from configs.methods.common import Common

class Method(Palavra, Common):
    emb_dim = 768
    checkpoint = Path(f'checkpoints/palavra-checkpoints/{emb_dim}')

    concepts_description: Literal['coarse', 'discriminative', 'rich'] = 'coarse'
    num_concepts = '[eval]cfg.Data.num_concepts'

    use_github_method = True
    personalize_token_images: Literal['all', 'fixed5','random5'] = "all"
    optimize_token = True

    load_concepts = None
    save_concepts = None
    load_image_pool = None
