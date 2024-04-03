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

class CIRCOBaseDataSet(Dataset):
    
    def __init__(self, image_transform=None, text_transform=None):
        self.root = Path(self.parent.dataset_root)
        self.split = self.root / 'annotations' / f'{self.parent.split}.json'
        
        self.image_transform = image_transform
        self.text_transform = text_transform

        self.max_imgs_to_lbls = 23
        
        self.df, self.paths = self._prepare_data()
        self.dl = self._prepare_dl()
                
    def _prepare_data(self):
        # Load COCO images information
        with open(self.root / 'COCO2017_unlabeled/annotations/image_info_unlabeled2017.json', 'r') as f_read:
            imgs_info = json.load(f_read)

        file_names = np.array(list(map(lambda img: img['file_name'], imgs_info['images'])))
        
        # Load Images Info
        paths = self.root / 'COCO2017_unlabeled/unlabeled2017' / file_names
        ids = [img_info["id"] for img_info in imgs_info["images"]]
        map_gts = {str(id): i for i, id in enumerate(ids)}

        # Load labels
        with open(self.split, "r") as f_read:
            annotations = json.load(f_read)
            
        df = pd.DataFrame.from_dict(annotations)

        if self.split.stem == 'val':
            df['gts'] = df['gt_img_ids'].apply(lambda x: [map_gts[str(i)] for i in x])
        df['reference_img_path'] = df['reference_img_id'].apply(lambda x: f'{x:012}.jpg')
        df['reference_img_id'] = list(range(len(df))) # the img_ids are unique
            
        return df, paths
        
    def _prepare_dl(self):
        return DataLoader(self, batch_size=self.parent.batch_size, num_workers=self.parent.num_worker)
    
    def process(self, line):
        raise NotImplementedError('Create a derived class and implement the method')
    
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):            
        return self.process(self.df.loc[idx])


class CIRCOEvalRetrievalDS(CIRCOBaseDataSet):

    def process(self, line):
        if self.split.stem == 'test':
            img = np.array([-1] * self.max_imgs_to_lbls)
        else:
            img = np.array(line['gts'] + [-1] * (self.max_imgs_to_lbls - len(line['gts'])))
        
        complete_caption = f"a photo of * that {line[f'relative_caption']}"
        label = (complete_caption  if self.text_transform is None else
                 self.text_transform(complete_caption))

        concepts_idx = line['reference_img_id']
        # f"a photo of $ that {caption}" for caption in relative_captions
        return img, label, np.array([concepts_idx])
        
        
    
class CIRCOLearnDS(CIRCOBaseDataSet):
    
    def process(self, line):

        img = Image.open(self.root / f'COCO2017_unlabeled/unlabeled2017/{line["reference_img_path"]}')
        img = (img if self.image_transform is None 
               else self.image_transform(img))
        
        label = (line[f'shared_concept']  if self.text_transform is None else
                 self.text_transform(line['shared_concept']))

        concept_id = line['reference_img_id']
        
        return img, label, np.array([concept_id])
        
    
class CIRCOImagePoolDS(CIRCOBaseDataSet):
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        img = (img if self.image_transform is None 
               else self.image_transform(img))
        return np.array(img), idx
    