# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

from typing import Literal
from datasets.fashioniq import FashionIQLearnDS, FashionIQImagePoolDS, FashionIQEvalRetrievalDS
from clearconf import BaseConfig
from collections import defaultdict

class FashionIQ(BaseConfig):
    dataset_root = 'datasets/fashion_iq'
    # label_type: Literal['0', '1'] = '0'  # NOTE two different set of captions, differences should be negligibles
    label_type = None
    split: Literal["shirt", "dress", "toptee"] = "shirt"
    split_type: Literal["train", "test", "val"] = "test"
    batch_size = 128
    num_worker = 4
    num_concepts:int = 5000  # NOTE the number of concepts is always less than this value
    create_dummy_missing = True

    coarse_concepts = defaultdict(lambda: FashionIQ.split)

    class LearnDS(FashionIQLearnDS): pass
    class ImagePoolDS(FashionIQImagePoolDS): pass
    class EvalDS(FashionIQEvalRetrievalDS): pass

    name = '[eval]f"{cls.mro()[0].__name__}_{cls.split}_{cls.label_type}"'
    
Data = FashionIQ
