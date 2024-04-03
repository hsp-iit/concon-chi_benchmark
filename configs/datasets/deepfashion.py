# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

from typing import Literal
from datasets.deepfashion import DeepFashionLearnDS, DeepFashionImagePoolDS, DeepFashionEvalRetrievalDS
from clearconf import BaseConfig
import pandas as pd
from pathlib import Path
import os

class DeepFashion(BaseConfig):
    dataset_root = Path('data/datasets/PerVLBenchmark')
    label_type: Literal['shortened', 'detailed'] = "detailed"  # NOTE default one are detailed, shortened will substitute them
    batch_size = 128
    num_worker = 4
    split: Literal['personalized_validation', 'personalized_test'] = "personalized_test"
    num_concepts:int = 50

    # Read all concepts
    all_coarse_concepts = {}
    captions = pd.read_csv(dataset_root / 'annotations' / 'deep_fashion' / 'train_coarse_grained_names.csv', sep=',')
    captions = captions.reset_index()
    for index, row in captions.iterrows():
        all_coarse_concepts[str(row['unique_pair_ids'])] = row['pair_id_categories']

    # Convert name of folder into coarse concepts
    coarse_concepts = []
    for pair_id in os.listdir(dataset_root / 'data' / 'deep_fashion2' / split / 'codes_infer'):
        coarse_concepts.append(all_coarse_concepts[pair_id])
    
    # To avoid to log them
    del index, row, captions, pair_id, all_coarse_concepts

    class LearnDS(DeepFashionLearnDS): pass
    class ImagePoolDS(DeepFashionImagePoolDS): pass
    class EvalDS(DeepFashionEvalRetrievalDS): pass

    name = '[eval]f"{cls.mro()[0].__name__}_{cls.split}_{cls.label_type}"'
    
Data = DeepFashion
