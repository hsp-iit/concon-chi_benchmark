# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

from clearconf import BaseConfig
from datasets.circo import CIRCOEvalRetrievalDS, CIRCOImagePoolDS, CIRCOLearnDS

class CIRCO(BaseConfig):
    dataset_root = 'data/datasets/CIRCO'
    batch_size = 128
    num_worker = 8
    split = 'val'
    
    num_concepts:int = 220
    
    class LearnDS(CIRCOLearnDS): pass
    class ImagePoolDS(CIRCOImagePoolDS): pass
    class EvalDS(CIRCOEvalRetrievalDS): pass
        
    name = '[eval]f"{cls.mro()[0].__name__}_{cls.split}"'
    
Data = CIRCO