# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, default_collate

def _collate_fn(batch):
    transposed = list(zip(*batch))
    res = []
    for samples in transposed:
        try:
            res += [default_collate(samples)]
        except TypeError:
             res += [samples]

    return res

class CirrBaseDataSet(Dataset):
    
    def __init__(self, image_transform=None, text_transform=None):
        self.root = Path(self.parent.dataset_root)
        self.split = self.parent.split
        
        self.image_transform = image_transform
        self.text_transform = text_transform

        self.max_imgs_to_lbls = 1
        
        self.df, self.paths = self._prepare_data()
        self.dl = self._prepare_dl()
                
    def _prepare_data(self):
        # Load COCO images information
        with open(self.root / 'captions' / f'cap.rc2.{self.split}.json', 'r') as f_read:
            annotations = json.load(f_read)
            
        with open(self.root / 'image_splits' / f'split.rc2.{self.split}.json', 'r') as f_read:
            imgs_info = json.load(f_read)

        paths = self.root / 'img_raw' / np.array(list(imgs_info.values()))
        df = pd.DataFrame(annotations)[['reference', 'caption']]

        if self.split != 'test1':
            df['gts'] = pd.DataFrame(annotations).target_hard.apply(lambda x: list(imgs_info).index(x))
            df['target_hard'] = pd.DataFrame(annotations).target_hard.apply(lambda x: imgs_info[x])
        
        df['reference'] = df['reference'].apply(lambda x: imgs_info[x])
        df['reference_img_id'] = pd.factorize(df['reference'])[0]

        return df, paths
        
    def _prepare_dl(self):
        return DataLoader(self, batch_size=self.parent.batch_size, num_workers=self.parent.num_worker, collate_fn=_collate_fn)
    
    def process(self, line):
        raise NotImplementedError('Create a derived class and implement the method')
    
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):            
        return self.process(self.df.loc[idx])


class CirrEvalRetrievalDS(CirrBaseDataSet):
    def __init__(self, image_transform=None, text_transform=None):
        super().__init__(image_transform, text_transform)

    def process(self, line):
        
        img = None
        if self.split != 'test1':
            img = np.array([line['gts']])
        
        complete_caption = f"* {line[f'caption']}"
        label = (complete_caption  if self.text_transform is None else
                 self.text_transform(complete_caption))

        concepts_idx = np.array([line['reference_img_id']])
        return img, label, concepts_idx
        
        
    
class CirrLearnDS(CirrBaseDataSet):
    
    def __init__(self, image_transform=None, text_transform=None):
        super().__init__(image_transform, text_transform)
        self.df = self.df.drop_duplicates('reference', ignore_index=True)
    
    def process(self, line):
        img = Image.open(self.root / 'img_raw' / line['reference'])
        img = (img if self.image_transform is None 
               else self.image_transform(img))

        concept_id = line['reference_img_id']
        
        return img, None, np.array([concept_id])
        
    
class CirrImagePoolDS(CirrBaseDataSet):
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        img = (img if self.image_transform is None 
               else self.image_transform(img))
        return img, idx
    
# class Aux(CirrEvalRetrievalDS):
#     class parent:
#         dataset_root = '/home/iit.local/arosasco/datasets/cirr'
#         split = 'val'
#         batch_size = 4
#         num_worker = 0
#     pass

# gt, labels, concepts = next(iter(Aux().dl))
# print(f'{gt=}')
# print(f'{labels=}')
# print(f'{concepts=}')