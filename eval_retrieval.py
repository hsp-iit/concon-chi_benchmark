# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

import argparse
from collections import defaultdict
from importlib import import_module
import inspect
import sys

import pandas as pd

from vlpers.utils.evaluation import get_ranks, mAP, mAP_at, mRR, recall_at, get_all_ranks
import vlpers.utils.logging as logging
from vlpers.utils.logging import logger
from vlpers.utils import misc
from pprint import pformat

from configs.retrieval import Config as cfg


def get_config():
    parser = argparse.ArgumentParser(prog='Eval Script')
    parser.add_argument('--method', choices=['pic2word', 'palavra', 'searle', 'baselines.image_only', 'baselines.random', 'baselines.text_only', 
                                             'baselines.image_text', 'baselines.concepts_only', 'baselines.context_only', 'baselines.concepts_context'], 
                                             help="Model to evaluate", required=True)
    parser.add_argument('--dataset', choices=['conconchi', 'circo', 'cirr', 'deepfashion', 'fashioniq'], 
                                             help="Dataset for evaluation", default='conconchi')
    
    args = parser.parse_args()
    sys.argv = sys.argv[0:1]    # Deleting arguments to avoid interference with other methods parseargs
    
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

    # Learn the new concepts embeddings
    concept_dataset = cfg.Data.LearnDS(image_transform=method.image_transform, text_transform=method.text_transform)
    if not cfg.Method.load_concepts:
        id = logger.progress(description='[red]Learning concepts...', total=len(concept_dataset.dl))
        method.set_mode(cfg.Method.Mode.LEARNING)
        for batch in concept_dataset.dl:
            images, _, concepts = batch 
            method.learn_concepts(images, concepts)
            logger.progress(id)

        if cfg.Logging.log_dir and cfg.Logging.save_concepts:
            method.save_concepts(cfg.Logging.exp_dir)

    # Populate image pool
    method.set_mode(cfg.Method.Mode.TESTING)
    image_dataset = cfg.Data.ImagePoolDS(image_transform=method.image_transform, text_transform=method.text_transform)
    if not cfg.Method.load_image_pool:
        id = logger.progress(description='[blue]Populating image pool...', total=len(image_dataset.dl))
        for batch in image_dataset.dl:
            images, *_ = batch 
            method.add_image_pool(images)
            logger.progress(id)
        if cfg.Logging.log_dir and cfg.Logging.save_image_pool:
            method.save_image_pool(cfg.Logging.exp_dir)

    # Retrieve images
    eval_dataset = cfg.Data.EvalDS(image_transform=method.image_transform, text_transform=method.text_transform)
    results = defaultdict(lambda:[])
    id = logger.progress(description='[green]Evaluating model...', total=len(eval_dataset.dl))
    for batch in eval_dataset.dl:
        gt, labels, concepts = batch

        scores = method.retrieve(descriptions=labels, concepts=concepts)

        # Log Metrics and log
        results['mRR'] += mRR(scores, gt, avg=False).tolist()
        results['mAP'] += mAP(scores, gt, avg=False).tolist()
        results['All Ranks'] += get_all_ranks(scores, gt).tolist()
        for k in [1, 5, 10]:
            results[f'R@{k}'] += recall_at(scores, gt, k=k, avg=False).tolist()
            results[f'mAP@{k}'] += mAP_at(scores, gt, k=k, avg=False).tolist()

        logger.progress(id)

    # Save Metrics 
    labels = eval_dataset.df
    results = pd.DataFrame.from_dict(results)
    results = pd.concat([labels, results], axis=1)
    
    if cfg.Logging.log_dir:
        misc.save_results(cfg.Logging.exp_dir, results)
        
    logger.info(f"\n{results[['mRR', 'mAP', 'mAP@1', 'mAP@5', 'mAP@10', 'R@1', 'R@5', 'R@10']].mean().to_frame().transpose().to_markdown()}\n")
    return results

if __name__ == '__main__':        
    try:
        main()
    except BaseException as e:
        logger.exception(e)
    finally:
        logger.progress(stop=True)