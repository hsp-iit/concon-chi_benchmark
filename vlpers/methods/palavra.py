# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

import numpy as np
from vlpers.utils.interface import RetrievalMethod
import torch
import clip
from palavra.caption_retrival_eval import _convert_image_to_rgb
from palavra.utils.non_nv import encode_text_with_learnt_tokens
from palavra.utils.deep_set_clf import D as deep_set
from palavra.utils.nv import TextVisualMap
import os
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
from simple_parsing import ArgumentParser
from palavra.caption_retrival_eval import EvalParams
from palavra.fsl_eval import infer_tokens_from_f_theta
from palavra.get_f_theta import HParams  # NECESSARY to load train configu
from palavra.fsl_eval import parse_coarse_grained_strings, optimize_token
import torch.nn.functional as F
import faiss
from vlpers.methods.common import encode_text_img_retrieval, get_concepts_pos
from palavra.get_f_theta import HParams 
import __main__
__main__.HParams = HParams

class Palavra(RetrievalMethod):
    
    def __init__(self, cfg=None) -> None:
        super().__init__()
        
        if cfg is None:
            cfg = self.__class__
        
        self.emb_dim: int = cfg.emb_dim
        self.image_pool = torch.zeros([0, self.emb_dim])
        self.num_concepts = self.parent.Data.num_concepts
        self.coarse_concepts = self.parent.Data.LearnDS().concepts_names[self.concepts_description]

        if cfg.load_concepts is not None:
            self.concepts_embeddings = torch.tensor(np.load(cfg.load_concepts), device=self.device)
        else:
            self.concepts_embeddings = torch.zeros([self.num_concepts, self.emb_dim])

        self.concepts_examples = torch.zeros([0, 3, 224, 224])
        self.concepts_ids = torch.zeros([0, 1], dtype=torch.long)
        self.index = faiss.IndexFlatL2(self.emb_dim)

        parser = ArgumentParser()
        parser.add_arguments(EvalParams, dest="evalparams")
        self.args, unknown = parser.parse_known_args()
        self.args.evalparams.model_path = cfg.checkpoint
        self.args.evalparams.is_optimize_token = cfg.optimize_token
        self.args.evalparams.is_short_captions = True
        self.args.evalparams.is_train_loader_no_reps = False
        self.args.evalparams.latent_ep = 30
        self.args.evalparams.random_seed = 5
        self.args.evalparams.set_size_override = 5
        self.args.evalparams.token_optimize_mode = 1

        self.args.personalize_token_images = cfg.personalize_token_images  # TODO TEST ######################
        
        if cfg.use_github_method:
            self.args.evalparams.is_coarse_grained_negative_per_class = False  # If true, it uses coarse grained as negative, otherwise it users only item
            self.args.evalparams.is_token_as_suffix = True
        else:
            self.args.evalparams.is_coarse_grained_negative_per_class = True  # If true, it uses coarse grained as negative, otherwise it users only item
            self.args.evalparams.is_token_as_suffix = False

        self.device = "cuda"
        train_args_name = [x for x in os.listdir(self.args.evalparams.model_path) if'args' in x][0]
        self.train_args = np.load(os.path.join(self.args.evalparams.model_path, train_args_name), allow_pickle = True)
        self.train_args = self.train_args.tolist()

        ######### Load tokenization model
        clip_model_name = "ViT-B/32" if self.emb_dim == 512 else "ViT-L/14"
        self.model, self.preprocess = clip.load(clip_model_name, device=self.device)
        logit_scale = self.model.logit_scale.exp()
        self.logit_scale = logit_scale.mean().cpu()

        #Inser CLIP text encoding with learnt token methods
        funcType = type(self.model.encode_text)
        self.model.encode_text_with_learnt_tokens = funcType(encode_text_with_learnt_tokens, self.model)
        self.model.end_id = 49407

        #Load deep set network (f_theta)
        self.set_model = deep_set(self.train_args.deep_set_d_dim, x_dim=self.emb_dim, out_dim = self.train_args.no_of_new_tokens*self.emb_dim, pool = 'mean', dropout=self.train_args.dropout)
        model_name = [x for x in os.listdir(self.args.evalparams.model_path) if'deep_set_model_' in x][0]

        self.set_model = self.set_model.to(self.device)
        self.set_model.load_state_dict(torch.load(os.path.join(self.args.evalparams.model_path,model_name)))
        self.set_model.eval()

        txt_vis_model_name = [x for x in os.listdir(self.args.evalparams.model_path) if'txt_vis_model_' in x][0]
        self.text_vis_model = TextVisualMap(emb_dim=self.emb_dim)
        self.text_vis_model.to(self.device)
        self.text_vis_model.load_state_dict(torch.load(os.path.join(self.args.evalparams.model_path,txt_vis_model_name)))
    
    @torch.no_grad()   
    def add_image_pool(self, images:torch.tensor, descriptions:torch.tensor=None):
        image_features = self.model.encode_image(images.cuda())
        image_features = F.normalize(image_features, dim=-1)
        self.image_pool = torch.vstack([self.image_pool, image_features.cpu().detach()])
        
    @torch.no_grad()   
    def empty_image_pool(self):
        self.image_pool = torch.zeros([0, self.emb_dim])
        self.index.reset()

    @torch.no_grad()
    def get_query_features(self, descriptions, concepts):
        embeddings = []
        for concepts_i, captions_i in zip(concepts, descriptions):

            text_concepts = self.concepts_embeddings[concepts_i]
            concepts_pos = get_concepts_pos(captions_i.unsqueeze(0), value=265, max_concepts=concepts.size(1)).squeeze(0)

            # SUB AXTERIX WITH COARSE GRAINED IF GITHUB, we go in opposite order in order to keep the true index of the asterix
            if self.args.evalparams.is_token_as_suffix: #with coarse grained (Github Method)
                for concept_id, concept_pos in zip(torch.flip(concepts_i[concepts_i != -1], dims=(-1,)), torch.flip(concepts_pos[0][concepts_pos[0] != 0], dims=(-1,))):
                    tokenized_coarse_grained = clip.tokenize(self.coarse_concepts[concept_id.item()])
                    tokenized_coarse_grained = tokenized_coarse_grained[tokenized_coarse_grained != 49406]
                    tokenized_coarse_grained = tokenized_coarse_grained[tokenized_coarse_grained != 49407]
                    tokenized_coarse_grained = tokenized_coarse_grained[tokenized_coarse_grained != 0]
                    captions_i = torch.cat([captions_i[..., :concept_pos], tokenized_coarse_grained, captions_i[..., concept_pos:]], dim=-1)[..., :captions_i.shape[-1]]
                concepts_pos = get_concepts_pos(captions_i.unsqueeze(0), value=265, max_concepts=concepts.size(1)).squeeze(0)  # Must do it again since position now are changed

            emb_for_retrive_phrase = encode_text_img_retrieval(self.model, captions_i.cuda().unsqueeze(0), text_concepts.unsqueeze(0).type(torch.half).cuda(), split_idxs=concepts_pos.cuda()).float()

            emb_for_retrive_phrase = F.normalize(emb_for_retrive_phrase, dim=-1)
            embeddings.append(emb_for_retrive_phrase)
        return torch.vstack(embeddings)
    
    @torch.no_grad()   
    def compute_similarities(self, text_features, image_features):
            return (self.logit_scale * image_features @ text_features.T).T

    def save_concepts(self, path):
        np.save(path, self.concepts_embeddings.cpu().numpy())
        
    def learn_concepts(self, concepts_examples, concepts_ids=None):
        # concepts_ids = torch.IntTensor(concepts_ids)
        # self.concepts_examples = torch.vstack([self.concepts_examples, torch.stack(concepts_examples)])
        self.concepts_examples = torch.vstack([self.concepts_examples, concepts_examples])
        self.concepts_ids = torch.vstack([self.concepts_ids, concepts_ids])

        for i in concepts_ids.unique():
            if i == -1:
                continue
            concept_ids = [str(i.item())]
            concept_examples = self.concepts_examples[self.concepts_ids[:, 0] == i]

            # Select how to optimize the token
            if self.args.personalize_token_images == "all":
                train_dataloader = [[concept_examples.unsqueeze(0), concept_ids]]
                reduced_concept_examples = concept_examples.clone()
            elif self.args.personalize_token_images in ["random5"]:
                train_dataloader = [[concept_examples.unsqueeze(0), concept_ids]]
                reduced_concept_examples = concept_examples[torch.randperm(len(concept_examples))[:5]]
            elif self.args.personalize_token_images in ["fixed5"]:
                reduced_concept_examples = concept_examples[torch.randperm(len(concept_examples))[:5]]
                train_dataloader = [[reduced_concept_examples.unsqueeze(0), concept_ids]]

            with torch.no_grad():
                image_features = self.model.encode_image(reduced_concept_examples.cuda())
                image_features = F.normalize(image_features, dim=-1)
                object_tokens = self.set_model(image_features.unsqueeze(0).float())

            if self.args.evalparams.is_optimize_token:
                gt_text_coarse = [self.coarse_concepts[i.item()]]
                gt_text_label_parsed = parse_coarse_grained_strings(None, gt_text_coarse)
                object_tokens = optimize_token(self.args, object_tokens, self.model, train_dataloader, gt_text_label_parsed, self.text_vis_model)
            self.concepts_embeddings[i] = object_tokens.squeeze().detach()

    @torch.no_grad()   
    def forget_concepts(self):
        self.concepts_embeddings = torch.zeros([self.num_concepts, self.emb_dim])
        self.concepts_examples = torch.zeros([0, 3, 224, 224])
        self.concepts_ids = torch.zeros([0, 3], dtype=torch.long)
    
    def retrieve(self, descriptions, concepts):   
        composed_features = self.get_query_features(descriptions, concepts)
        return self.compute_similarities(composed_features.cpu(), self.image_pool).detach()
        
    def train(self):
        pass
        
    def eval(self):
        pass
    
    def image_transform(self, x):
        x = self.preprocess(x)
        return x

    def text_transform(self, x):
        x.LABEL = clip.tokenize(x.LABEL)[0]
        return x