# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

from typing import Literal
from vlpers.datasets.conconchi import CCCEvalRetrievalDS, CCCImagePoolDS, CCCLearnDS
from clearconf import BaseConfig

class ConConChi(BaseConfig):
    dataset_root = 'datasets/ConCon-Chi'
    batch_size = 128
    num_worker = 8
    
    num_concepts:int = 20
    class LearnDS(CCCLearnDS):
        split = 'train.json'
    class ImagePoolDS(CCCImagePoolDS):
        split = 'test.json'
    class EvalDS(CCCEvalRetrievalDS):
        split = 'test.json'

    name = '[eval]f"{cls.mro()[0].__name__}_{cls.LearnDS.split[:-5]}"'
    
Data = ConConChi

