# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

from vlpers.methods.searle import SEARLE
from configs.methods.common import Common

class Method(SEARLE, Common):
    num_concepts = '[eval]cfg.Data.num_concepts'

    load_concepts = None
    load_image_pool = None
        

