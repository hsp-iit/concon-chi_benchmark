# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

import numpy as np
from PIL import Image
from vlpers.utils.interface import RetrievalMethod
from vlpers.utils.logging import logger
import torch
from diffusers import DiffusionPipeline

class DiffusersProgress:

    def __init__(self,description):
        super().__init__()
        self.description = description
        self.id = None
        
    def __call__(self, total):
        self.id = logger.progress(description=self.description, total=total)
        return self
    
    def update(self):
        logger.progress(self.id)
    
    def __enter__(self):
        return self
    def __exit__(self, *args, **kwargs):
        logger.progress(self.id, visible=False)

#TODO set seed

class SDM(RetrievalMethod):
    
    def __init__(self, cfg=None) -> None:
        super().__init__()

        self.pipeline = DiffusionPipeline.from_pretrained(
            self.pretrained_model_name_or_path,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        self.pipeline.to(self.device)
        self.concepts_names = self.parent.Data.LearnDS().concepts_names[self.concepts_description]
        
    def learn_concepts(self, concepts_examples, concepts_ids=None):        
        pass


    @torch.no_grad()
    def generate(self, batch):
        descriptions, concepts = batch['INPUT'], batch['CONCEPTS']
        if len(concept := np.unique(concepts)) != 1:
            print('SDM can handle one personalized concept at a time')
            exit()

        self.pipeline.progress_bar = DiffusersProgress('[red]Generating images')
        descriptions = np.repeat(descriptions, self.samples_per_label, axis=0).tolist()
        images = self.pipeline(descriptions, num_inference_steps=self.inference_steps, guidance_scale=self.guidance, 
                               negative_prompt=self.negative_prompt*len(descriptions)).images
        images = [images[i:i+self.samples_per_label] for i in range(0, len(images), self.samples_per_label)]
        
        return images
    
    def load(self, batch):
        descriptions, concepts_idx = batch['LABEL'], batch['CONCEPTS']
        if len(concept_idx := np.unique(concepts_idx)) != 1:
            raise ValueError('SDM can process one concept at a time')
        
        paths = self.cached_img_path / str(concept_idx.item()) / descriptions
        images = []
        for p in paths:
            images += [[Image.open(img) for img in p / np.array([f'img_{i}.jpg' for i in range(self.samples_per_label)])]]
               
        return images
    
    def train(self):
        pass
        
    def eval(self):
        pass

    def image_transform(self, x):
        return x

    def train_text_transform(self, x):
        res = np.full([77], '_', dtype='<U39')
        res[:len(x.split())] = np.array(x.split())
       
        return res
    
    def eval_text_transform(self, ds, x):
        x['INPUT'] = np.array(x.LABEL.replace('*', f'{self.concepts_names[x.CONCEPTS]}', 1))
        
        x.GTS = np.array(x.GTS + [-1] * (ds.max_gts - len(x.GTS)))
        x.LABEL = np.array(x.LABEL)
        x.CONCEPTS = np.array([x.CONCEPTS])

        return x