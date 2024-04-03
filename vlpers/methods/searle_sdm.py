# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

from pathlib import Path
import gc
import re
from typing import Any, Optional
import numpy as np
import clip
from clip import tokenize
from PIL import Image
from datasets.conconchi import coarse_concepts, pattern
from datasets.conconchi import concepts
from methods.common import get_concepts_pos
from utils.interface import RetrievalMethod
import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from utils.logging import logger
from transformers import CLIPTextModel
from torch import nn

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

class SearleSDM(RetrievalMethod):
    
    def __init__(self, cfg=None) -> None:
        super().__init__()

        self.pipeline = DiffusionPipeline.from_pretrained(
            self.pretrained_model_name_or_path,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        self.pipeline.to(self.device)
        
        self.model, self.preprocess_val = clip.load("ViT-L/14", device=self.device, jit=False)
        self.model.end_id = self.model.vocab_size - 1
        self.model.float()
        
        if self.concepts_path is not None:
            self.concepts_embeddings = torch.tensor(np.load(self.concepts_path / 'concepts.npy'), device=self.device)
            
        
        old_dict = self.pipeline.text_encoder.text_model.embeddings.token_embedding.weight
        self.new_offset = old_dict.shape[0]
        
        new_dict = torch.concatenate([old_dict, self.concepts_embeddings.to(torch.float16)])
        token_embedding = nn.Embedding(*new_dict.shape)
        token_embedding.weight.data = new_dict
        
        self.pipeline.text_encoder.text_model.embeddings.token_embedding = token_embedding
        self.i = 0
        
    def learn_concepts(self, concepts_examples, concepts_ids=None):        
        pass


    @torch.no_grad()
    def generate(self, batch):
        batch['CONCEPTS'] = [torch.tensor(self.i) for _ in range(len(batch['CONCEPTS']))]
        descriptions, concepts = batch['INPUT'], batch['CONCEPTS']
        if len(concept := np.unique(concepts)) != 1:
            print('SDM can handle one personalized concept at a time')
            exit()
            
        descriptions[descriptions == 265] = concept.item() + self.new_offset
        descriptions = descriptions.repeat_interleave(repeats=4, dim=0)
        
        text_features = self.pipeline.text_encoder(descriptions.cuda())[0]

        self.pipeline.progress_bar = DiffusersProgress('[red]Generating images')
        descriptions = np.repeat(descriptions, self.samples_per_label, axis=0).tolist()
        images = self.pipeline(prompt_embeds=text_features, num_inference_steps=self.inference_steps, guidance_scale=self.guidance, 
                               ).images
        images = [images[i:i+self.samples_per_label] for i in range(0, len(images), self.samples_per_label)]
        
        self.i += 1
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
        x['INPUT'] = tokenize('*')[0]
        
        x.GTS = np.array(x.GTS + [-1] * (ds.max_gts - len(x.GTS)))
        x.LABEL = np.array(x.LABEL)
        x.CONCEPTS = torch.tensor([x.CONCEPTS])

        return x
