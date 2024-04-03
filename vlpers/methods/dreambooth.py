# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

import gc
from typing import Any, Optional
import numpy as np
from PIL import Image
import torch

from vlpers.utils.interface import RetrievalMethod
from vlpers.utils.logging import logger

from diffusers import StableDiffusionPipeline
from diffusers.train_dreambooth import parse_args, main

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

class DreamBooth(RetrievalMethod):
    
    def __init__(self) -> None:
        super().__init__()
        self.args = parse_args()
        vars(self.args).update(self.Diffusers.to_dict())

        concepts_names = self.parent.Data.LearnDS().concepts_names
        self.concepts = concepts_names['concepts']
        self.coarse_concepts  = concepts_names['coarse']

        self.last_concept = None
        
    def learn_concepts(self, concepts_examples, concepts_ids=None):        
        # concepts_examples = concepts_examples.to(self.device)
        concepts_ids = torch.tensor(concepts_ids) # .to(self.device)
        
        if not torch.all(concepts_ids[:, 1:] == -1):
            raise ValueError("Only one concept per image is supported")
        
        concepts_id = torch.unique(concepts_ids[:, 0])
        if not len(concepts_id) == 1:
            raise ValueError("Only single concept batches are supported")
        
        concept_name = self.concepts[concepts_id.item()]
        coarse_name  = self.coarse_concepts[concepts_id.item()]
        
        train_concept_dir = (self.parent.Logging.exp_dir / 'concept_images' / concept_name)
        train_concept_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(concepts_examples):
            img_path = train_concept_dir / f'{i}_{concept_name}.{img.format.lower()}'
            if not img_path.exists():
                img.save(img_path)
        
        self.args.instance_data_dir=train_concept_dir.as_posix()
        self.args.class_data_dir=(self.class_data_dir / coarse_name.replace(' ', '_')).as_posix()
        self.args.output_dir=(self.parent.Logging.exp_dir / 'checkpoints' / concept_name.replace(' ', '_')).as_posix()
        self.args.instance_prompt=f"a photo of sks {coarse_name}"
        self.args.class_prompt=f"a photo of {coarse_name}"
        
        self.args.max_train_steps = self.train_steps[concepts_id.item()]
        self.args.checkpointing_steps = self.train_steps[concepts_id.item()]
        self.args.validation_prompt = [f'a photo of sks {coarse_name} on a beach', f'a photo of sks {coarse_name} on the moon', f'a photo of sks {coarse_name} with a cat', f'a photo of a yellow sks {coarse_name}']
        self.args.validation_steps = self.train_steps[concepts_id.item()] # 50
        self.args.num_validation_images = 4
        
        main(self.args)
        torch.cuda.empty_cache()
        gc.collect()

    @torch.no_grad()
    def generate(self, batch):
        descriptions = batch['INPUT']
        
        if len(concept := np.unique(batch['CONCEPTS'])) != 1:
            print('Dreambooth can handle one personalized concept at a time')
            exit()
            
        if self.last_concept != self.concepts[concept.item()]:
            
            if self.concepts_path is not None:
                concept_ckpt = self.concepts_path / self.concepts[concept.item()]
            else:
                concept_ckpt = self.parent.exp_dir / 'checkpoints' / self.concepts[concept.item()]
                
            pipeline = StableDiffusionPipeline.from_pretrained(str(concept_ckpt), torch_dtype=torch.float16).to('cuda')
            pipeline.safety_checker = None
            # pipeline.set_progress_bar_config(disable=True)
            pipeline.progress_bar = DiffusersProgress('[red]Generating images')
            self.pipeline = pipeline
            
        self.last_concept = self.concepts[concept.item()]
            
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

    def train_text_transform(self, ds, x):
        res = np.full([77], '_', dtype='<U39')
        res[:len(x.split())] = np.array(x.split())
       
        return res

    def eval_text_transform(self, ds, x):
        x['INPUT'] = np.array(x.LABEL.replace('*', f'sks {self.coarse_concepts[x.CONCEPTS]}', 1))
        
        x.GTS = np.array(x.GTS + [-1] * (ds.max_gts - len(x.GTS)))
        x.LABEL = np.array(x.LABEL)
        x.CONCEPTS = np.array([x.CONCEPTS])

        return x