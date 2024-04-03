# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

import warnings
import numpy as np
import clip
from vlpers.methods.common import encode_text_img_retrieval, get_concepts_pos
from vlpers.utils.interface import RetrievalMethod
from pandas.errors import SettingWithCopyWarning
import torch

class ClipBaseline(RetrievalMethod):

    def __init__(self) -> None:
        super().__init__()
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        self.model.float()

        logit_scale = self.model.logit_scale.exp()
        self.logit_scale = logit_scale.mean().cpu()

        self.image_pool = torch.zeros([0, 768])
        self.concepts_embeddings = torch.zeros([self.num_concepts, 768], device=self.device)
        
        if self.load_image_pool is not None:
            self.image_pool = torch.tensor(np.load(self.load_image_pool), device='cpu')

    @torch.no_grad()
    def add_image_pool(self, images:torch.tensor, descriptions:torch.tensor=None):
        images = images.to(self.device)
        image_features = self.model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        self.image_pool = torch.vstack([self.image_pool, image_features.cpu()])

    def save_concepts(self, path):
        np.save(path / 'concepts.npy', self.concepts_embeddings.cpu().numpy())
        
    def save_image_pool(self, path):
        np.save(path / 'image_pool.npy', self.image_pool.cpu().numpy())

    def retrieve(self, descriptions, concepts):   
        composed_features = self.get_query_features(descriptions, concepts)
        return self.compute_similarities(composed_features, self.image_pool).detach()
        
    @torch.no_grad()
    def compute_similarities(self, text_features, image_features):
        text_features = text_features.cpu()
        return (self.logit_scale * image_features @ text_features.T).T

    def image_transform(self, x):
        x = self.preprocess(x)
        return x

    def text_transform(self, x): return x
    
    def set_mode(self, mode: RetrievalMethod.Mode):
        super().set_mode(mode)
        Mode = RetrievalMethod.Mode
        match mode:
            case Mode.LEARNING | Mode.TESTING:
                self.model.eval()
            case Mode.TRAINING:
                self.method.train()

    def get_query_features(self, descriptions, concepts): pass
    def learn_concepts(self, concepts_examples, concepts_ids=None): pass


class ImageText(ClipBaseline):

    def __init__(self) -> None:
        super().__init__()

        self.concepts_embeddings_sum = torch.zeros([self.num_concepts, 768], device=self.device)
        self.concepts_embeddings_count = torch.zeros([self.num_concepts], device=self.device)
        
        if self.load_concepts is not None:
            self.concepts_embeddings = torch.tensor(np.load(self.load_concepts / 'concepts.npy'), device=self.device)

        self.concepts_names = [''] * 20

    @torch.no_grad()
    def learn_concepts(self, concepts_examples, concepts_ids=None):
        concepts_examples =concepts_examples.to(self.device)
        concepts_ids = concepts_ids.to(self.device)

        if not torch.all(concepts_ids[:, 1:] == -1):
            raise ValueError("Learn one concept at a time")

        embeddings = self.model.encode_image(concepts_examples)

        self.concepts_embeddings_count += (concepts_ids[:, 0:1] == torch.arange(self.num_concepts, device=concepts_ids.device)).sum(dim=0)
        embeddings = (embeddings / embeddings.norm(dim=-1, keepdim=True))
        self.concepts_embeddings_sum = self.concepts_embeddings_sum.index_add(0, concepts_ids[:, 0], embeddings)
        self.concepts_embeddings = self.concepts_embeddings_sum / self.concepts_embeddings_count[:, None]
        self.concepts_embeddings = self.concepts_embeddings / self.concepts_embeddings.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def get_query_features(self, descriptions, concepts):
        concepts = concepts.to(self.device)
        descriptions = descriptions.to(self.device)

        # Compute Image Features 
        # (If there are more than one concepts in the same description their embeddings are averaged)
        image_features = self.concepts_embeddings[concepts]
        image_features[concepts == -1] = 0.
        image_features = (image_features.sum(1) / (concepts != -1).sum(dim=-1, keepdim=True))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Compute Text Features
        text_features = self.model.encode_text(descriptions).to(self.image_pool.dtype)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Weighted Average
        features = (1 - self.alpha) * image_features + (self.alpha) * text_features
        features = features / features.norm(dim=-1, keepdim=True)
        return features
        

    def text_transform(self, x):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

            for c in x.CONCEPTS:
                x.LABEL = x.LABEL.replace('*', f'{self.concepts_names[c]}', 1)
            x.LABEL = clip.tokenize(x.LABEL)[0]
        
        return x
    
    

class TextOnly(ImageText):
    def __init__(self) -> None:
        super().__init__()
        self.concepts_names = self.parent.Data.LearnDS().concepts_names[self.concepts_description]
        self.alpha = 1


class ImageOnly(ImageText):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = 0
        self.concepts_names = [''] * self.num_concepts
