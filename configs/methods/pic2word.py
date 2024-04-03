# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

from pathlib import Path
from configs.methods.common import Common
from vlpers.methods.pic2word import Pic2Word

class Method(Pic2Word, Common):
    checkpoint = Path('checkpoints/pic2word_model.pt')
    num_concepts = '[eval]cfg.Data.num_concepts'
    
    load_concepts = None
    load_image_pool = None
