# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
from torch.utils.data import DataLoader, Dataset
from utils.logging import logger
import pandas as pd
import numpy as np


def load_captions(data_path, split, split_type, label_type, print_info=False, create_dummy_missing=False):
    data_path = data_path / split
    try:
        captions_path = f'datasets/fashion-iq/captions/cap.{split}.{split_type}.json'
        with open(captions_path) as infile:
            entries = json.load(infile)
    except:
        raise("Clone the fashion-iq repository in datasets!")

    labels = []
    concepts = []
    images = []
    missing = 0
    for elem in entries:
        if not create_dummy_missing:
            exists_candidate = os.path.exists(data_path / f"{elem['candidate']}.jpg")
            exists_target = os.path.exists(data_path / f"{elem['target']}.jpg")
            if not exists_candidate or not exists_target:
                missing += 1
                continue    
        labels.append(set((f"{split} {elem['captions'][0]}", f"{split} {elem['captions'][1]}")))
        concepts.append(f"{elem['candidate']}.jpg")
        if 'target' in elem:
            images.append(f"{elem['target']}.jpg")
        else:
            images.append("DUMMY.jpg")
    if print_info:
        logger.info(f"Missing {missing} entries over a total of {len(entries)} (either 'target' or 'candidate' is missing from that label, these labels will be ignored)")

    # LOAD IMAGE POOL
    pool_path = f'datasets/fashion-iq/image_splits/split.{split}.{split_type}.json'
    with open(pool_path) as infile:
        image_pool = json.load(infile)
    image_pool = [f"{x}.jpg" for x in image_pool]
    if not create_dummy_missing:
        missing = 0
        filtered = []
        for image in image_pool:
            if os.path.exists(f"data/datasets/fashion_iq/images/{split}/{image}"):
                filtered.append(image)
            else:
                missing += 1
        if print_info:
            logger.info(f"Missing {missing} entries for image pool over a total of {len(image_pool)}")
        image_pool = filtered


    return labels, concepts, images, image_pool


class FashionIQImagePoolDS(Dataset):
    
    def __init__(self, image_transform=lambda x:x, text_transform=lambda x:x):
        self.root = Path(self.parent.dataset_root)
        self.data_path = self.root / 'images'
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.split_type = self.parent.split_type
        self.create_dummy_missing = self.parent.create_dummy_missing

        self.labels, self.concepts_idx, self.images, self.image_pool = load_captions(self.data_path, self.parent.split, self.split_type, self.parent.label_type, print_info=True, create_dummy_missing=self.create_dummy_missing)
        self.data_path = self.data_path / self.parent.split

        self.dl = DataLoader(self, batch_size=self.parent.batch_size, num_workers=self.parent.num_worker)
        self.df = None

    def __len__(self):
        return len(self.image_pool)

    def __getitem__(self, idx):
        if os.path.exists(self.data_path / self.image_pool[idx]):
            img = Image.open(self.data_path / self.image_pool[idx])   
        elif self.create_dummy_missing:
            img = np.random.rand(240, 240, 3)
        else:
            raise Exception(f"Img {self.data_path / self.image_pool[idx]} cannot be found")
        img = self.image_transform(img)
        return img, [], []


class FashionIQEvalRetrievalDS(Dataset):
    
    def __init__(self, image_transform=lambda x:x, text_transform=lambda x:x):
        self.root = Path(self.parent.dataset_root)
        self.data_path = self.root / 'images'
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.split_type = self.parent.split_type
        self.create_dummy_missing = self.parent.create_dummy_missing

        self.labels, self.concepts_idx, self.images, self.image_pool = load_captions(self.data_path, self.parent.split, self.split_type, self.parent.label_type, print_info=True, create_dummy_missing=self.create_dummy_missing)
        self.data_path = self.data_path / self.parent.split

        self.concepts = list(dict.fromkeys(self.concepts_idx))

        self.dl = DataLoader(self, batch_size=self.parent.batch_size, num_workers=self.parent.num_worker)
        self.df = pd.DataFrame({"LABEL": self.labels, "CONCEPTS": list(map(self.concepts.index, self.concepts_idx)), "GT": list(map(self.image_pool.index, self.images))})
        self.df.LABEL = self.df.LABEL.apply(lambda x: set((list(x)[i].replace(self.parent.split, ' * ', 1) for i in range(len(x)))))
        self.df.CONCEPTS = self.df.CONCEPTS.apply(lambda x: [x])
        self.df.GT = self.df.GT.apply(lambda x: [x])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Prepare image list
        img = torch.tensor(self.df.loc[idx].GT)
        img = torch.cat([img, torch.full([3 - len(img)], -1, dtype=torch.long)]) # pad to reach length 3
        
        # Prepare concepts
        index = self.df.loc[idx].CONCEPTS
        concepts_idx = torch.tensor(index, dtype=torch.int64) # associate index to concepts
        concepts_idx = torch.cat([concepts_idx, torch.full([3 - len(concepts_idx)], -1, dtype=torch.long)]) # pad to reach length 3

        # Prepare labels
        # lbl = self.df.loc[idx].LABEL
        # lbl = lbl.replace(self.parent.split, ' * ')
        lbl = self.df.loc[idx].LABEL
        if len(lbl) > 1:
            lbl = ' and '.join(lbl)
        else:
            lbl = list(lbl)[0]

        if lbl.count('*') > 1:
            lbl = lbl[::-1].replace('*', '', 1)[::-1]

        line = self.df.loc[idx].copy()
        line.LABEL = lbl
        lbl = self.text_transform(line).LABEL

        return img, lbl, concepts_idx

class FashionIQLearnDS(Dataset):
    
    def __init__(self,image_transform=lambda x:x, text_transform=lambda x:x):
        self.root = Path(self.parent.dataset_root)
        self.data_path = self.root / 'images'
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.split_type = self.parent.split_type
        self.create_dummy_missing = self.parent.create_dummy_missing

        self.labels, self.concepts_idx, self.images, _ = load_captions(self.data_path, self.parent.split, self.split_type, self.parent.label_type, print_info=True, create_dummy_missing=self.create_dummy_missing)
        self.data_path = self.data_path / self.parent.split

        self.concepts = list(dict.fromkeys(self.concepts_idx))

        self.dl = DataLoader(self, batch_size=None, num_workers=self.parent.num_worker)
        self.df = None
            
    def __len__(self):
        return len(self.concepts)

    def __getitem__(self, idx):
        if len(self) <= idx:
            raise StopIteration()
        
        concept_ids = []
        processed_images = []
        
        # Process image
        if os.path.exists(self.data_path / self.concepts[idx]):
            img = Image.open(self.data_path / self.concepts[idx])   
        elif self.create_dummy_missing:
            img = np.random.rand(240, 240, 3)
        else:
            raise Exception(f"Img {self.data_path / self.concepts[idx]} cannot be found")
        img = self.image_transform(img)
        concept_ids.append([idx])
        processed_images += [img]

        return torch.stack(processed_images), None, torch.tensor(concept_ids)
