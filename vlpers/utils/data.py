# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

from torch.utils.data.dataloader import default_collate
import numpy as np

def collate_fn(batch):
    elem = batch[0]
    
    if isinstance(elem, tuple):
    
        transposed = list(zip(*batch))
        res = []
        for samples in transposed:
            try:
                res += [default_collate(samples)]
            except (TypeError, RuntimeError):
                # an np string array is not processed by default_collate but 
                # can still be packed in an np array so...
                if isinstance(samples[0], np.ndarray):
                    samples = np.array(samples)

                res += [samples]
                
    elif isinstance(elem, dict):
        res = {}
        for key in elem:
            try:
                samples = [d[key] for d in batch]
                samples = default_collate(samples)
                
            except (TypeError, RuntimeError):
                if isinstance(samples[0], np.ndarray):
                    samples = np.array(samples)

            res[key] = samples

    return res