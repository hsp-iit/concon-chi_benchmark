# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

from pathlib import Path
import numpy as np
from datasets.conconchi import concepts as concepts_names
from utils.interface import RetrievalMethod
import torch
import subprocess
import os
from PIL import Image
import shutil

#TODO set seed
class LDM(RetrievalMethod):
    
    def __init__(self, cfg=None) -> None:
        super().__init__()

        self.cfg = self.get_cfg()
        self.txt2img_args = self.cfg.Txt2ImgArgs.to_dict()
        self.exp_dir = Path(self.cfg.parent.Logging.exp_dir)
        self.concepts_names = self.cfg.concepts_names

    @staticmethod
    def convert_args(args):
        list_args = []
        for key, value in args.items():
            if len(key) == 1:
                list_args.append(f"-{key}")
            else:
                list_args.append(f"--{key}")
            value = str(value)
            if len(value) > 0:
                list_args.append(value)
        return list_args
        
    def learn_concepts(self, concepts_examples, concepts_ids=None):
        pass

    def generate(self, descriptions, concepts):

        all_gen_images = []

        for desc, conc in zip(descriptions, concepts):

            desc = np.array(desc).copy()
            desc = " ".join(desc[desc != '_'].tolist())
            desc = desc.replace('*', self.concepts_names[conc.item()])
            conc = concepts_names[conc]
            desc = f"\"{desc}\""  # f"'a photo of {desc}'"
            # print(desc)

            self.txt2img_args["embedding_path"] = "logs/TI_TRAIN_1_VEC/alienser2023-10-11T17-56-34_alienser/checkpoints/embeddings_gs-99.pt"
            self.txt2img_args["prompt"] = desc
            self.txt2img_args["outdir"] = self.exp_dir / conc
            
            list_args = self.convert_args(self.txt2img_args)

            subprocess.call("/home/iit.local/sberti/mambaforge/envs/ldm/bin/python third_party/textual_inversion/scripts/txt2img.py " + ' '.join(list_args), shell=True , stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

            gen_images = []
            for i in range(len(os.listdir(self.exp_dir / conc / 'samples'))):
                gen_images.append(Image.open(self.exp_dir / conc / 'samples' / f"{i:04d}.jpg"))
            assert i > 0, "Could not generate images from subprocess"
            all_gen_images.append(gen_images)

            shutil.rmtree(self.exp_dir / conc)

        return all_gen_images
    
    def load(self, descriptions, concepts):
        images = []
        for description, concept in zip(descriptions, concepts):
            path = self.cached_img_path / str(concept.item()) / description
            images += [[Image.open(img) for img in path / np.array([f'img_{i}.jpg' for i in range(self.samples_per_label)])]]
        
        return images
    
    @torch.no_grad()
    def get_query_features(self, descriptions, concepts):
        descriptions = descriptions.to(self.device)
        concepts = concepts.to(self.device)
        
    @torch.no_grad()
    def compute_similarities(self, text_features, image_features):
        pass
    
    def train(self):
        pass
        
    def eval(self):
        pass

    def image_transform(self, x):
        return x

    def eval_text_transform(self, x):
        return x
    
    def train_text_transform(self, x):
        return x
