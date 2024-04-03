# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

import copy
import json
from pathlib import Path
from itertools import accumulate
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from PIL import Image

from vlpers.utils.logging import logger
from vlpers.utils.data import collate_fn

class VlpersDataset(Dataset):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        

class CCCBaseDataset(VlpersDataset):
    num_concepts:int = 20
    concepts_names = None
    
    def __init__(self, image_transform=lambda x:x, text_transform=lambda x:x, **kwargs):
        super().__init__(image_transform=image_transform, text_transform=text_transform, **kwargs)
        
        self.root = Path(self.parent.dataset_root)
        self.split = self.root / 'data' / 'annotations' / self.split
        self.num_worker = self.parent.num_worker
        self.batch_size = self.parent.batch_size
        
        with (self.split).open('r') as f:
            annotations = json.load(f)
        
        self.concepts_names = pd.DataFrame(annotations['concepts'])
        self.max_gts = annotations['max_gts']
        self.num_concepts = len(self.concepts_names)
        self.max_concepts = annotations['max_concepts']
        self.annotations = annotations
        
        self.df = self._prepare_data(annotations)
        self.dl = self._prepare_dl()
        self.i = 0
        
    def _prepare_data(self, annotations):
        return pd.DataFrame(annotations['data']).reset_index(drop=True)
        
    def _prepare_dl(self):
        return DataLoader(self, batch_size=self.batch_size, num_workers=self.num_worker, collate_fn=collate_fn)
        
    def process(self, line):
        raise NotImplementedError('Create a derived class and implement the method')
                
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):            
        return self.process(self.df.loc[idx])


class CCCImagePoolDS(CCCBaseDataset):

    def _prepare_data(self, annotations):
        df = super()._prepare_data(annotations)
        df =  df.explode('GTS').reset_index(drop=True)
        df['ID_GTS'] = df.index
        return df

    def process(self, line):
        line = self.text_transform(copy.deepcopy(line))

        img = Image.open(self.root / 'data' / 'images' / line['GTS'])
        img = (img if self.image_transform is None 
               else self.image_transform(img))
        
        label = line.LABEL

        concept_id = line['CONCEPTS']
        concept_id = np.array(concept_id + [-1] * (self.max_concepts - len(concept_id)))
        
        return img, label, concept_id


class CCCEvalRetrievalDS(CCCBaseDataset):
    
    def _prepare_data(self, annotations):
        df = super()._prepare_data(annotations)
        
        aux = df.explode('GTS').reset_index(drop=True)
        aux['ID_GTS'] = aux.index
        aux = aux[['GTS', 'ID_GTS']].set_index('GTS')
        self.map = aux
        
        df = df[df['KIND'] != 'negative'].reset_index(drop=True)
        
        # Some additional GTS might refer to images that are not in the image_pool 
        # (because they are in the training set or in another exclusive image_pool)
        df['ID_GTS'] = (df['GTS'] + df['ADDITIONAL GTS']).apply(lambda gts: [aux.loc[gt]['ID_GTS'] for gt in gts if gt in aux.index])
        
        return df

    def process(self, line):
        line = self.text_transform(copy.deepcopy(line))

        gts = line['ID_GTS']
        gts = np.array(gts + [-1] * (self.max_gts - len(gts)))
        
        label = line.LABEL
        
        concept_id = line['CONCEPTS']
        concept_id = np.array(concept_id + [-1] * (self.max_concepts - len(concept_id)))
        
        return gts, label, concept_id
    

class CCCLearnDS(CCCBaseDataset):

    def __init__(self, image_transform=lambda x:x, text_transform=lambda x:x):
        super().__init__(image_transform, text_transform)
        if self.max_concepts != 1:
            raise ValueError('The training split should contain exactly one concept per example')
        
    def _prepare_data(self, annotations):
        df = super()._prepare_data(annotations)
        return df.explode('GTS').reset_index(drop=True)

    def process(self, line):
        line = self.text_transform(copy.deepcopy(line))
        
        img = Image.open(self.root / 'data' / 'images' / line['GTS'])
        img = (img if self.image_transform is None 
               else self.image_transform(img))
        
        label = line.LABEL

        concept_id = line['CONCEPTS']
        concept_id = torch.tensor(concept_id)
        
        return img, label, concept_id
    
    def __len__(self):
        return self.num_concepts

    def __getitem__(self, idx):
        concept_df = self.df[self.df['CONCEPTS'].apply(lambda x: x[0]) == idx]
        batch = collate_fn([self.process(line) for i, line in concept_df.iterrows()])
        
        return batch
        
    def _prepare_dl(self):
        return DataLoader(self, batch_size=None, num_workers=self.num_worker)


class GenerationEvalDatasetCCC(CCCBaseDataset):

    def __init__(self, image_transform=lambda x:x, text_transform=lambda x:x, **kwargs):
        super().__init__(image_transform=image_transform, text_transform=text_transform, **kwargs)
        
        concepts_len = self.df.CONCEPTS.value_counts().sort_index().to_list()
        self.concepts_batch_len = [(c_len // self.batch_size) + ((c_len % self.batch_size) != 0) for c_len in concepts_len]
        self.cumulative_batch_len = np.array(list(accumulate(self.concepts_batch_len)))
    
    def process(self, line):
        line = self.text_transform(self, line)
        return line.to_dict()

    def __getitem__(self, idx):
        concept_idx = np.argwhere(self.cumulative_batch_len > idx).min()
        mb_idx = (idx - (self.cumulative_batch_len[concept_idx - 1] if concept_idx > 0 else 0))     

        df = self.df[self.df.CONCEPTS == concept_idx]
        df = df.iloc[mb_idx * self.batch_size: ((mb_idx + 1) * self.batch_size)]

        batch = collate_fn([self.process(line) for i, line in df.iterrows()])

        return batch
        
    def _prepare_dl(self):
        return DataLoader(self, batch_size=None, num_workers=self.num_worker)
    
    def __len__(self):
        return sum(self.concepts_batch_len)
    
    def _prepare_data(self, annotations):
        df =  super()._prepare_data(annotations)
        # remove all queries that don't have exactly one concept
        df = df[df.CONCEPTS.apply(len) == 1].reset_index(drop=True)
        # unpack CONCEPTS (gives error if lists have more or less than one item)
        df.CONCEPTS =  df.CONCEPTS.apply(lambda x: (lambda x: x)(*x))
        df['GTS'] = df['GTS'] + df['ADDITIONAL GTS']
        return df.sort_values(by='CONCEPTS').reset_index(drop=True)