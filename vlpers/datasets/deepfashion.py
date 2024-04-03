# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, default_collate
from torch.utils.data import Dataset
from PIL import Image
from utils.logging import logger
import os


class DeepFashionImagePoolDS(Dataset):
    
    def __init__(self, image_transform=lambda x:x, text_transform=lambda x:x):
        self.image_transform = image_transform
        self.text_transform = text_transform
        
        self.root = Path(self.parent.dataset_root)  # PerVLBenckmark       
        split = Path(self.parent.split)  # personalized_validation

        self.concepts_dir = self.root / 'data' / 'deep_fashion2' / split / 'codes_infer'
        self.concepts = os.listdir(self.concepts_dir)
        self.pattern = rf"\b({'|'.join(map(re.escape, self.concepts))})\b"
            
        self.image_pool_dir = self.root / 'data' / 'deep_fashion2' / split / 'eval' / 'test'
        cap_name = 'validation_captions.csv' if str(split) == "personalized_validation" else "test_captions.csv"
        annotations = pd.read_csv(self.root / 'annotations' / 'deep_fashion' / cap_name, sep=',')
        images = annotations['image_name'].to_list()
        captions = annotations['caption'].to_list()
        pair_id = annotations['pair_id'].to_list()
        self.labels = []
        self.concepts_idx = []
        self.images = []
        for image, caption, id in zip(images, captions, pair_id):
            self.labels.append(caption.replace("person", str(id)))
            self.concepts_idx.append(str(id))
            self.images.append(image.split('/')[-1])

        self.dl = DataLoader(self, batch_size=self.parent.batch_size, num_workers=self.parent.num_worker)

            
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.image_pool_dir / self.images[idx])     
        img = self.image_transform(img)
        return img, [], []


class DeepFashionEvalRetrievalDS(Dataset):
    
    def __init__(self, image_transform=lambda x:x, text_transform=lambda x:x):
        self.image_transform = image_transform
        self.text_transform = text_transform

        self.root = Path(self.parent.dataset_root)
        self.label_type = self.parent.label_type

        self.concepts = os.listdir(self.root / 'data' / 'deep_fashion2' / self.parent.split / 'codes_infer')
        self.pattern = rf"\b({'|'.join(map(re.escape, self.concepts))})\b"

        cap_name = 'validation_captions.csv' if self.parent.split == "personalized_validation" else "test_captions.csv"
        annotations = pd.read_csv(self.root / 'annotations' / 'deep_fashion' / cap_name, sep=',')
        real_annotations_path = f"{self.label_type}_deepfashion2_captions.csv"
        real_annotations = pd.read_csv(self.root / 'annotations' / 'deep_fashion' / real_annotations_path, sep=',')

        df = annotations.set_index('image_name').join(real_annotations.set_index('image_name'), lsuffix='_caller', rsuffix='_other')
        df = df.rename(columns={"caption_other": "LABEL", 'pair_id_caller': 'CONCEPTS'}).drop(columns=['caption_caller', 'pair_id_other'])
        df.LABEL = df.LABEL.apply(lambda x: x.replace('person', ' * ', 1))
        df.CONCEPTS = df.CONCEPTS.apply(lambda x: str(x))
        df.CONCEPTS = df.CONCEPTS.apply(lambda x: [self.concepts.index(x)])
        df.reset_index(inplace=True)
        df.image_name = df.index

        self.dl = DataLoader(self, batch_size=self.parent.batch_size, num_workers=self.parent.num_worker)
        self.df = df

    def __getitem__(self, idx):            
        return self.process(self.df.loc[idx])

            
    def __len__(self):
        return len(self.df.index)

    def process(self, line):
        # Prepare image list
        img = line.image_name
        img = torch.tensor([img])
        
        # Prepare concepts
        concepts_idx = torch.tensor(line.CONCEPTS, dtype=torch.int64) # associate index to concepts
        concepts_idx = torch.cat([concepts_idx, torch.full([3 - len(concepts_idx)], -1, dtype=torch.long)]) # pad to reach length 3

        # Prepare labels
        line = self.text_transform(line)
        lbl = line.LABEL

        return img, lbl, concepts_idx

class DeepFashionLearnDS(Dataset):
    
    def __init__(self, image_transform=lambda x:x, text_transform=lambda x:x):
        self.image_transform = image_transform
        self.text_transform = text_transform
        
        self.root = Path(self.parent.dataset_root)
        self.split = Path(self.parent.split)    
        self.num_concepts = self.parent.num_concepts
        
        self.concepts_dir = self.root / 'data' / 'deep_fashion2' / self.split / 'codes_infer'
        self.concepts = os.listdir(self.concepts_dir)
        self.num_concepts = len(self.concepts)
        self.pattern = rf"\b({'|'.join(map(re.escape, self.concepts))})\b"

        self.dl = DataLoader(self, batch_size=None, num_workers=self.parent.num_worker)
            
    def __len__(self):
        return self.num_concepts

    def __getitem__(self, idx):
        if len(self) <= idx:
            raise StopIteration()
        
        concept_ids = []
        processed_images = []

        for img in os.listdir(self.concepts_dir / self.concepts[idx]):
        
            # Process image
            img = Image.open(self.concepts_dir / self.concepts[idx] / img)
            img = self.image_transform(img)
            concept_ids.append([idx])
            processed_images += [img]

        return torch.stack(processed_images), None, torch.tensor(concept_ids)
