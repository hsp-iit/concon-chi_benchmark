# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

import sys
import argparse
from collections import defaultdict
from importlib import import_module
import inspect
from pprint import pformat

import pandas as pd
import numpy as np

from vlpers.utils import misc
from vlpers.datasets.conconchi import GenerationEvalDatasetCCC
from vlpers.utils.evaluation import CLIPImageConcept, DensityCoverageContext, CLIPTextContext
import vlpers.utils.logging as logging
from vlpers.utils.logging import logger
from configs.generation import Config as cfg


def get_config():
    parser = argparse.ArgumentParser(prog='Eval Script')
    parser.add_argument('--method', choices=['dreambooth', 'textual_inversion', 'baselines.ldm', 'baselines.sdm', 'baselines.searle_sdm'], help="Model to evaluate", required=True)
    parser.add_argument('--dataset', choices=['conconchi'], help="Dataset for evaluation", default='conconchi')
    
    args = parser.parse_args()
    # Deleting arguments to avoid interference with other methods parseargs
    sys.argv = sys.argv[0:1]
    
    logger.info(f'Loading configuration: configs.{args.method} and configs.datasets.{args.dataset}')

    cfg.Data = import_module(f'configs.datasets.{args.dataset}').Data
    cfg.Method = import_module(f'configs.methods.{args.method}').Method

    return cfg


def main():    
    cfg = get_config()
    misc.set_reproducibility()
    
    if cfg.Logging.log_dir:
        cfg.Logging.exp_dir = (cfg.Logging.log_dir / cfg.Logging.exp_dir)
        cfg.Logging.exp_dir.mkdir(parents=True, exist_ok=True)
        
        logging.enable_file_handler(cfg.Logging.exp_dir / 'logs.txt')
        misc.save_config(cfg.Logging.exp_dir, cfg)
        misc.git_check_workspace(make_patch=cfg.Git.make_patch, path=cfg.Logging.exp_dir)
    else:
        logger.warning('Log dir not set. Checkpoints and results won\'t be saved.')
    
    logger.info(f'\n{pformat(cfg.to_dict())}\n')
    logger.info(inspect.getmodule(cfg))
    
    method = cfg.Method()
        
    class parent:
        dataset_root = 'datasets/ConCon-Chi'
        batch_size = 4
        num_worker = 8

    eval_dataset = GenerationEvalDatasetCCC( 
                            text_transform=method.eval_text_transform,
                            split='test.json',
                            parent=parent)

    concept_dataset = cfg.Data.LearnDS(image_transform=method.image_transform)
    if cfg.Method.concepts_path is None and cfg.Method.cached_img_path is None:
        # Learn the new concepts embeddings
        id = logger.progress(description='[red]Learning concepts...', total=len(concept_dataset.dl))
        for batch in concept_dataset.dl:
            images, _, concepts = batch
            
            method.learn_concepts(images, concepts)  
            logger.progress(id)

    # Retreive images
    results = defaultdict(lambda:[])
    
    get_images = method.generate if cfg.Method.cached_img_path is None else method.load
    
    clip_I_concept = CLIPImageConcept(cfg.Data.LearnDS())
    clip_T_context = CLIPTextContext()
    density_coverage = DensityCoverageContext(eval_dataset, k=10)

    id = logger.progress(description='[blue]Evaluating model...', total=len(eval_dataset.dl))
    for batch in eval_dataset.dl:
        gt, labels, concepts =\
            batch['GTS'], batch['LABEL'], batch['CONCEPTS']

        gen_images = get_images(batch) # descriptions=labels, concepts=concepts
        
        context_score = clip_T_context(batch, generations=gen_images) # gt=labels
        concept_score = clip_I_concept(batch, generations=gen_images) # gt=concepts
        density, coverage = density_coverage(batch, generations=gen_images)

        # # Log Metrics and log
        
        results['Context'] += context_score
        results['Concept'] += concept_score
        results['Density'] += density
        results['Coverage'] += coverage
        
        if cfg.Logging.save_images:
            misc.save_images(paths=cfg.Logging.exp_dir/'eval_samples'/np.array([f'{c}/{l}' for c, l in zip(concepts, labels)]),
                            images=gen_images)
        logger.progress(id)

    # Save Metrics 
    results = pd.DataFrame.from_dict(results)
    results = pd.concat([eval_dataset.df, results], axis=1)
    if cfg.Logging.log_dir: misc.save_results(cfg.Logging.exp_dir, results)

    logger.info(f"\n{results[['Context', 'Concept', 'Density', 'Coverage']].mean().to_frame().transpose().to_markdown()}\n")


if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        logger.exception(e)
    finally:
        logger.progress(stop=True)