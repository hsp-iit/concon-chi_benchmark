# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

from clearconf import BaseConfig
from datasets.cirr import CirrEvalRetrievalDS, CirrImagePoolDS, CirrLearnDS

class Cirr(BaseConfig):
    dataset_root = 'data/datasets/CIRR'
    batch_size = 128
    num_worker = 8
    split = 'val'
    
    num_concepts:int = 2165
    
    class LearnDS(CirrLearnDS): pass
    class ImagePoolDS(CirrImagePoolDS): pass
    class EvalDS(CirrEvalRetrievalDS): pass
        
    name = '[eval]f"{cls._name}_{cls.split}"'
    
Data = Cirr