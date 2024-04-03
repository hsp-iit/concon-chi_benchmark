# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

import copy
import gc
from pathlib import Path
import re
import numpy as np
import torch
import sklearn.metrics
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from collections import defaultdict

import vlpers.datasets.conconchi as ds
from vlpers.utils.logging import logger
from vlpers.utils.prdc import compute_prdc

def get_all_ranks(logits_matrix, ground_truth):
    # The full value exceed the highes value possible 
    preds = torch.full_like(ground_truth, -2, dtype=torch.long)
    
    # return the indexes corresponging to the computed scores
    # ranking[0][2] -> 6 than the image number 6 is the third most similar to label 0 
    all_ranks = torch.argsort(logits_matrix, descending=True)
    
    return all_ranks.numpy()

def get_ranks(logits_matrix, ground_truth):
    # The full value exceed the highes value possible 
    preds = torch.full_like(ground_truth, -2, dtype=torch.long)
    
    # return the indexes corresponging to the computed scores
    # ranking[0][2] -> 6 than the image number 6 is the third most similar to label 0 
    all_ranks = torch.argsort(logits_matrix, descending=True)
    # Row wise equality returns a boolean matrix with True where the correct images for
    # each labels are
    ground_truth_mask = (all_ranks.unsqueeze(2) == ground_truth.unsqueeze(1)).sum(dim=-1).bool()
    count = ground_truth_mask.sum(dim=1)
    
    unfolded_preds = torch.where(ground_truth_mask)[1]
    unfolded_preds = unfolded_preds.detach().cpu()
    
    range_tensor = torch.arange(preds.size(1))[None].repeat(preds.size(0), 1)
    mask = range_tensor < count.unsqueeze(1)
    
    preds[mask] = unfolded_preds
    preds = preds + 1
    
    # Finding the gts corresponding to the ranks
    img_ids = all_ranks[torch.arange(all_ranks.shape[0])[..., None], preds - 1]
    img_ids[~mask] = -1

    return preds.numpy(), img_ids.numpy()

def recall_at(logits_matrix, ground_truth, k=1, avg=True):
    ranks, _ = get_ranks(logits_matrix, ground_truth)
    ranks = copy.deepcopy(ranks).astype(np.float32)
    ranks[ranks == -1] = np.nan
    
    res = np.any(ranks <= k, axis=-1)
    
    if avg:
        res = np.mean(res)
    return res

def mRR(logits_matrix, ground_truth, avg=True):
    # Mean Reciprocal Rank
    ranks, _ = get_ranks(logits_matrix, ground_truth)
    ranks = copy.deepcopy(ranks).astype(np.float32)
    ranks[ranks == -1] = np.nan
    
    res = 1 / np.nanmin(ranks, axis=1)
    
    if avg:
        res = np.mean(res)
    return res

def mAP(logits_matrix, ground_truth, avg=True):
    # Mean Average Precisin
    ranks, _ = get_ranks(logits_matrix, ground_truth)
    ranks = copy.deepcopy(ranks).astype(np.float32)
    
    ranks[ranks == -1] = np.nan
    
    res = np.nanmean(np.arange(1, ranks.shape[1] + 1) / ranks, axis=1)
    
    if avg:
        res = np.mean(res)
    return res


def mAP_at(logits_matrix, ground_truth, k=50, avg=True):
    # Mean Average Precisin
    ranks, _ = get_ranks(logits_matrix, ground_truth)
    ranks = copy.deepcopy(ranks).astype(np.float32)
    ranks[ranks == -1] = np.nan
    
    gts_count = (~np.isnan(ranks)).sum(axis=1)
    ranks[ranks > k] = np.nan
    
    precision = np.arange(1, ranks.shape[1] + 1) / ranks
    res = np.nansum(precision, axis=1) / np.clip(gts_count, 0, k)
    
    if avg:
        res = np.mean(res)
    return res


model = None
processor = None

def get_clip():
    global model, processor
    if model is None:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", device_map='cuda')
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", device_map='cuda')
    return model, processor


class CLIPImageConcept:
    def __init__(self, dataset) -> None:        
        self.model, self.processor = get_clip()
        
        concepts_embeddings = []
    
        for batch in dataset.dl:
            images, _, concepts = batch 
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            with torch.no_grad():
                embedding = self.model.get_image_features(**inputs) 
                embedding /= embedding.norm(dim=-1, keepdim=True)
                embedding = embedding.mean(dim=0, keepdim=True)
                embedding /= embedding.norm(dim=-1, keepdim=True)
            
            concepts_embeddings += [embedding]
            
        self.concepts_embeddings = torch.concat(concepts_embeddings)

    def __call__(self, batch, generations):
        res = []

        if len(concept := np.unique(batch['CONCEPTS'])) != 1:
            print('The samples in a minibatch have to contain the same concept. I\'m lazy.')
            exit()

        concept_embedding = self.concepts_embeddings[concept.item()]

        for g_imgs in generations:
            inputs = self.processor(images=g_imgs, return_tensors="pt", padding=True)
            gen_embeddings = self.model.get_image_features(**inputs) 
            gen_embeddings = gen_embeddings / gen_embeddings.norm(dim=-1, keepdim=True)

            similarity = torch.matmul(concept_embedding, gen_embeddings.t())
            res += [similarity.mean().item()]
        
        return res
        
        
class CLIPTextContext:
    def __init__(self) -> None:
        self.model, self.processor = get_clip()
        
    def __call__(self, batch, generations):
        res = []
        gt = [label.replace('*', '') for label in batch['LABEL']]
        
        for gt, imgs in zip(gt, generations):
            with torch.no_grad():
                input = self.processor(text=gt, images=imgs,return_tensors="pt", padding=True)
                
                score = self.model(**input)['logits_per_text'].cuda() / self.model.logit_scale.exp()
                
                res += [score.mean().item()]
                
        return res

class DensityCoverage:
    
    def __init__(self, dataset_root):    
        self.model, self.processor = get_clip()
        self.dataset_root = Path(dataset_root)
        self.k = 3
    
    def __call__(self, batch, generations):
        density = []
        coverage = []

        pil_real = list(map(lambda x: [Image.open(self.dataset_root / 'data/images' / img) for img in x if img != '-1'], batch['GTS']))
        for r_imgs, g_imgs in zip(pil_real, generations):
            if len(r_imgs) < self.k + 2:
                density += [np.nan]
                coverage += [np.nan]
                continue

            imgs = r_imgs + g_imgs

            with torch.no_grad():
                input = self.processor(images=imgs, return_tensors="pt", padding=True)
                embeddings = self.model.get_image_features(**input)

            real_features = embeddings[:len(r_imgs)]
            real_features /= real_features.norm(dim=-1, keepdim=True)

            fake_features = embeddings[len(r_imgs):]
            fake_features /= fake_features.norm(dim=-1, keepdim=True)
            
            res = compute_prdc(real_features.cpu(), fake_features.cpu(), self.k)
            density += [res['density']]
            coverage += [res['coverage']]
            
        return density, coverage
    

class DensityCoverageContext:

    def __init__(self, dataset, k=3):    
        self.model, self.processor = get_clip()

        self.k = k
        self.real_features = defaultdict(lambda: torch.zeros([0, self.model.projection_dim]))

        id = logger.progress(description='[red]Initializing Density/Coverage...', total=len(dataset.dl))
        for batch in dataset.dl:
            gt, labels, concepts =\
                batch['GTS'], batch['LABEL'], batch['CONCEPTS']

            imgs = [Image.open(Path(dataset.parent.dataset_root) / 'data/images' / img) for imgs in batch['GTS'] for img in imgs if img != '-1']
            with torch.no_grad():
                input = self.processor(images=imgs, return_tensors="pt", padding=True)
                embeddings = self.model.get_image_features(**input).cpu()
                embeddings /= embeddings.norm(dim=-1, keepdim=True)

            self.real_features[np.unique(batch['CONCEPTS']).item()] =\
                torch.concatenate([self.real_features[np.unique(batch['CONCEPTS']).item()], embeddings])
            
            logger.progress(id)

        logger.info('DC initialized')

    def __call__(self, batch, generations):
        density = []
        coverage = []

        concept_id = np.unique((batch['CONCEPTS'])).item()

        for g_imgs in generations:

            with torch.no_grad():
                input = self.processor(images=g_imgs, return_tensors="pt", padding=True)
                fake_features = self.model.get_image_features(**input)

            fake_features /= fake_features.norm(dim=-1, keepdim=True)

            res = compute_prdc(self.real_features[concept_id].cuda(), fake_features.cuda(), self.k)
            density += [res['density']]
            coverage += [res['coverage']]

        return density, coverage
