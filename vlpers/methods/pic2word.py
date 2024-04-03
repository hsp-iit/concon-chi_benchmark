# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

import numpy as np
from pic2word.third_party.open_clip.clip import tokenize
from pic2word.eval_retrieval import load_model
from pic2word.params import parse_args
from vlpers.methods.common import encode_text_img_retrieval, get_concepts_pos
from vlpers.utils.interface import RetrievalMethod
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, Pad
from PIL import Image
import torchvision.transforms.functional as F

import warnings

class Pic2Word(RetrievalMethod):
    
    def __init__(self) -> None:
        super().__init__()
          
        args = parse_args()
        args.resume=self.checkpoint
        args.gpu = None if self.device == 'cpu' else int(self.device[-1])
        args.model="ViT-L/14"
        
        self.model, self.img2text, self.preprocess_val = load_model(args)
        logit_scale = self.model.logit_scale.exp()
        self.logit_scale = logit_scale.mean().cpu()
        
        self.image_pool = torch.zeros([0, 768])
        
        self.concepts_embeddings_sum = torch.zeros([self.num_concepts, 768], device=self.device)
        self.concepts_embeddings_count = torch.zeros([self.num_concepts], device=self.device)
        self.concepts_embeddings = None
        
        if self.load_concepts is not None:
            self.concepts_embeddings = torch.tensor(np.load(self.load_concepts / 'concepts.npy'), device=self.device)
            
        if self.load_image_pool is not None:
            self.image_pool = torch.tensor(np.load(self.load_image_pool), device='cpu')
        
        self.eval()
    
    @torch.no_grad()
    def add_image_pool(self, images:torch.tensor, descriptions:torch.tensor=None):
        images = images.to(self.device)
        
        image_features = self.model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        self.image_pool = torch.vstack([self.image_pool, image_features.cpu()])
        
    @torch.no_grad()
    def empty_image_pool(self):
        self.image_pool = torch.zeros([0, 768])
        
    @torch.no_grad()
    def learn_concepts(self, concepts_examples, concepts_ids=None):
        concepts_examples = concepts_examples.to(self.device)
        concepts_ids = concepts_ids.to(self.device)
        
        if not torch.all(concepts_ids[:, 1:] == -1):
            raise ValueError("More concepts per image are not supported.")
        
        query_image_features = self.model.encode_image(concepts_examples)
        query_image_tokens = self.img2text(query_image_features)
        
        self.concepts_embeddings_count += (concepts_ids[:, 0:1] == torch.arange(self.num_concepts, device=concepts_ids.device)).sum(dim=0)
        self.concepts_embeddings_sum = self.concepts_embeddings_sum.index_add(0, concepts_ids[:, 0], query_image_tokens)
        self.concepts_embeddings = self.concepts_embeddings_sum / self.concepts_embeddings_count[:, None]
        
    @torch.no_grad()
    def forget_concepts(self):
        self.concepts_embeddings = None
        self.concepts_embeddings_sum = torch.zeros([self.num_concepts, 768], device="cuda")
        self.concepts_embeddings_count = torch.zeros([self.num_concepts], device="cuda")
        
    def save_concepts(self, path):
        np.save(path / 'concepts.npy', self.concepts_embeddings.cpu().numpy())
        
    def save_image_pool(self, path):
        np.save(path / 'image_pool.npy', self.image_pool.cpu().numpy())
    
    def retrieve(self, descriptions, concepts):   
        composed_features = self.get_query_features(descriptions, concepts)
        return self.compute_similarities(composed_features.cpu(), self.image_pool).detach()
    
    @torch.no_grad()
    def get_query_features(self, descriptions, concepts):
        descriptions = descriptions.to(self.device)
        concepts = concepts.to(self.device)
        
        concepts_embeddings = self.concepts_embeddings[concepts]
        concepts_pos = get_concepts_pos(descriptions, value=265, max_concepts=concepts.size(1))
        composed_features = encode_text_img_retrieval(self.model, descriptions, concepts_embeddings, split_idxs=concepts_pos)                        
        composed_features = composed_features / composed_features.norm(dim=-1, keepdim=True)
        
        return composed_features
        
    @torch.no_grad()
    def compute_similarities(self, text_features, image_features):
        return (self.logit_scale * image_features @ text_features.T).T
    
    def train(self):
        self.model.train()
        self.img2text.train()
        
    def eval(self):
        self.model.eval()
        self.img2text.eval()

    def image_transform(self, x):        
        x = self.preprocess_val(x)
        return x

    def text_transform(self, x):
        x.LABEL = tokenize(x.LABEL)[0]
        return x