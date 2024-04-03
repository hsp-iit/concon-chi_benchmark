# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

from pathlib import Path
import gc
import numpy as np
from vlpers.utils.interface import RetrievalMethod
import torch
import subprocess
import os
from PIL import Image
import shutil
from clearconf import BaseConfig
# from vlpers.datasets.conconchi import single_token_coarse_concepts as coarse_concepts

#TODO set seed
class TextualInversion(RetrievalMethod, BaseConfig):
    
    def __init__(self, cfg=None) -> None:
        super().__init__()

        self.coarse_concepts = self.parent.Data.LearnDS().concepts_names["coarse"]
        self.concepts_names = self.parent.Data.LearnDS().concepts_names["discriminative"]

        self.cfg = self.get_cfg()
        self.learn_tokens_args = self.cfg.LearnTokensArgs.to_dict()
        self.txt2img_args = self.cfg.Txt2ImgArgs.to_dict()
        self.exp_dir = Path(self.cfg.parent.Logging.exp_dir)
        self.personalized_embeddings = {}
        self.base_model = self.cfg.base_model
        self.ldm_env_interpreter_path = self.cfg.ldm_env_interpreter_path
        self.print_debug = self.cfg.print_debug

        # If we already know the concepts, we load them directly
        if self.cfg.concepts_path != None:
            for concept_name in self.concepts_names:
                concept_folder = list(filter(lambda x: concept_name in x, os.listdir(self.cfg.concepts_path)))
                if len(concept_folder) == 1:
                    if self.checkpoints_selection_method == "fixed":
                        checkpoint = self.checkpoint
                    elif self.checkpoints_selection_method == "auto":
                        checkpoint = self.automatic_steps[concept_name]
                    self.personalized_embeddings[concept_name] = os.path.join(self.cfg.concepts_path, concept_folder[0], f'checkpoints/embeddings_gs-{checkpoint}.pt')
                else:
                    raise Exception(f"Missing {self.embedding_path}")

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
        concepts_ids = torch.tensor(concepts_ids)
        
        if not torch.all(concepts_ids[:, 1:] == -1):
            raise ValueError("Only one concept per image is supported")
        
        concepts_id = torch.unique(concepts_ids[:, 0])
        if not len(concepts_id) == 1:
            raise ValueError("Only single concept batches are supported")
        
        concept_name = self.concepts_names[concepts_id.item()]
        coarse_name  = self.coarse_concepts[concepts_id.item()]
        
        train_concept_dir = (self.exp_dir / 'concept_images' / concept_name)
        train_concept_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(concepts_examples):
            img_path = train_concept_dir / f'{i}_{concept_name}.{img.format.lower()}'
            if not img_path.exists():
                img.save(img_path)
        
        self.learn_tokens_args["data_root"] = train_concept_dir.as_posix()
        self.learn_tokens_args["init_word"] = coarse_name
        self.learn_tokens_args["l"] = self.exp_dir
        self.learn_tokens_args["n"] = concept_name
        
        list_args = self.convert_args(self.learn_tokens_args)
        list_args.append("--no-test")

        if self.print_debug:
            subprocess.call(f"{self.ldm_env_interpreter_path} vlpers/third_party/methods/textual_inversion/main.py " + ' '.join(list_args), shell=True)
        else:
            subprocess.call(f"{self.ldm_env_interpreter_path} vlpers/third_party/methods/textual_inversion/main.py " + ' '.join(list_args), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        # Add folder to personalized_embeddings
        try:
            concept_folder = list(filter(lambda x: concept_name in x, list(os.listdir(os.path.join(self.exp_dir)))))[0]
        except Exception as e:
            raise Exception("Something went wrong when calling the ldm env with textual inversion. Did you download the checkpoint?")

        self.personalized_embeddings[concept_name] = os.path.join(self.exp_dir, concept_folder, f'checkpoints/embeddings_gs-{self.checkpoint}.pt')

        torch.cuda.empty_cache()
        gc.collect()

    def generate(self, descriptions, concepts):

        all_gen_images = []

        for desc, conc in zip(descriptions, concepts):

            desc = np.array(desc).copy()
            desc = " ".join(desc[desc != '_'].tolist())
            # desc = desc.replace(conc, " * ")
            # conc = concepts_names[concepts_names.index(conc)]
            conc = concepts_names[conc]
            desc = f"\"{desc}\""  # f"'a photo of {desc}'"

            self.txt2img_args["embedding_path"] = self.personalized_embeddings[conc]
            self.txt2img_args["prompt"] = desc
            self.txt2img_args["outdir"] = self.exp_dir / conc
            
            list_args = self.convert_args(self.txt2img_args)

            if self.base_model == 'ldm':
                subprocess.call(f"{self.ldm_env_interpreter_path} vlpers/third_party/textual_inversion/scripts/txt2img.py " + ' '.join(list_args), shell=True , stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            elif self.base_model == 'sdm':
                subprocess.call(f"{self.ldm_env_interpreter_path} vlpers/third_party/textual_inversion/scripts/stable_txt2img.py " + ' '.join(list_args), shell=True , stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            

            gen_images = []
            for i in range(len(os.listdir(self.exp_dir / conc / 'samples'))):
                gen_images.append(Image.open(self.exp_dir / conc / 'samples' / f"{i:05d}.jpg"))
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
