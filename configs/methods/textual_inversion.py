# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

from pathlib import Path
from typing import Literal
from vlpers.methods.textual_inversion import TextualInversion
from configs.methods.common import Common
import os

base_model: Literal['ldm', 'sdm'] = 'ldm'

class Method(TextualInversion, Common):
    concepts_path = None
    checkpoint = 4999
    cached_img_path = None
    samples_per_label = 4
    checkpoints_selection_method: Literal['auto', 'fixed'] = "auto"
    base_model = base_model

    num_concepts = '[eval]cfg.Data.num_concepts'

    print_debug = True
    ldm_env_interpreter_path = f"{os.path.expanduser('~')}/mambaforge/envs/ldm/bin/python"
    if not os.path.exists(ldm_env_interpreter_path):
        ldm_env_interpreter_path = f"{os.path.expanduser('~')}/conda/envs/ldm/bin/python"
    if not os.path.exists(ldm_env_interpreter_path):
        raise Exception("To test textual_inversion, you need to install the ldm environment!")

    # FOR LDM
    automatic_steps = {
        "mybox": 2099,
        "redbag": 3899,
        "mymdevice": 2699,
        "toomouse": 2699,
        "myclown": 5299,
        "bottluminum": 3899,
        "mydoll": 5699,
        "octobag": 3099,
        "mybottle": 5299,
        "myshirt": 299,
        "snox": 99,
        "goldensweat": 3699,
        "alienser": 4899,
        "bluebag": 2099,
        "birdyer": 4299,
        "heartyshirt": 1699,
        "mybird": 899,
        "mysprayer": 2099,
        "mydispenser": 899,
        "mymplush": 4099
    }

    # FOR SDM
    automatic_steps = {
        "mybox": 4999,
        "redbag": 4999,
        "mymdevice": 4999,
        "toomouse": 4999,
        "myclown": 4999,
        "bottluminum": 4999,
        "mydoll": 4999,
        "octobag": 4999,
        "mybottle": 4999,
        "myshirt": 4999,
        "snox": 4999,
        "goldensweat": 4999,
        "alienser": 4999,
        "bluebag": 4999,
        "birdyer": 4999,
        "heartyshirt": 4999,
        "mybird": 4999,
        "mysprayer": 4999,
        "mydispenser": 4999,
        "mymplush": 4999
    }
    
    class LearnTokensArgs:
        '''
        usage: main.py [-h] [-n [NAME]] [-r [RESUME]] [-b [base_config.yaml [base_config.yaml ...]]] [-t [TRAIN]] [--no-test [NO_TEST]] [-p PROJECT] [-d [DEBUG]] [-s SEED] [-f POSTFIX] [-l LOGDIR] [--scale_lr [SCALE_LR]] [--datadir_in_name [DATADIR_IN_NAME]] --actual_resume ACTUAL_RESUME
            --data_root DATA_ROOT [--embedding_manager_ckpt EMBEDDING_MANAGER_CKPT] [--placeholder_string PLACEHOLDER_STRING] [--init_word INIT_WORD] [--logger [LOGGER]] [--checkpoint_callback [CHECKPOINT_CALLBACK]] [--enable_checkpointing [ENABLE_CHECKPOINTING]]
            [--default_root_dir DEFAULT_ROOT_DIR] [--gradient_clip_val GRADIENT_CLIP_VAL] [--gradient_clip_algorithm GRADIENT_CLIP_ALGORITHM] [--process_position PROCESS_POSITION] [--num_nodes NUM_NODES] [--num_processes NUM_PROCESSES] [--devices DEVICES] [--gpus GPUS]
            [--auto_select_gpus [AUTO_SELECT_GPUS]] [--tpu_cores TPU_CORES] [--ipus IPUS] [--log_gpu_memory LOG_GPU_MEMORY] [--progress_bar_refresh_rate PROGRESS_BAR_REFRESH_RATE] [--enable_progress_bar [ENABLE_PROGRESS_BAR]] [--overfit_batches OVERFIT_BATCHES]
            [--track_grad_norm TRACK_GRAD_NORM] [--check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH] [--fast_dev_run [FAST_DEV_RUN]] [--accumulate_grad_batches ACCUMULATE_GRAD_BATCHES] [--max_epochs MAX_EPOCHS] [--min_epochs MIN_EPOCHS] [--max_steps MAX_STEPS]
            [--min_steps MIN_STEPS] [--max_time MAX_TIME] [--limit_train_batches LIMIT_TRAIN_BATCHES] [--limit_val_batches LIMIT_VAL_BATCHES] [--limit_test_batches LIMIT_TEST_BATCHES] [--limit_predict_batches LIMIT_PREDICT_BATCHES] [--val_check_interval VAL_CHECK_INTERVAL]
            [--flush_logs_every_n_steps FLUSH_LOGS_EVERY_N_STEPS] [--log_every_n_steps LOG_EVERY_N_STEPS] [--accelerator ACCELERATOR] [--strategy STRATEGY] [--sync_batchnorm [SYNC_BATCHNORM]] [--precision PRECISION] [--enable_model_summary [ENABLE_MODEL_SUMMARY]]
            [--weights_summary WEIGHTS_SUMMARY] [--weights_save_path WEIGHTS_SAVE_PATH] [--num_sanity_val_steps NUM_SANITY_VAL_STEPS] [--resume_from_checkpoint RESUME_FROM_CHECKPOINT] [--profiler PROFILER] [--benchmark [BENCHMARK]] [--deterministic [DETERMINISTIC]]
            [--reload_dataloaders_every_n_epochs RELOAD_DATALOADERS_EVERY_N_EPOCHS] [--reload_dataloaders_every_epoch [RELOAD_DATALOADERS_EVERY_EPOCH]] [--auto_lr_find [AUTO_LR_FIND]] [--replace_sampler_ddp [REPLACE_SAMPLER_DDP]] [--detect_anomaly [DETECT_ANOMALY]]
            [--auto_scale_batch_size [AUTO_SCALE_BATCH_SIZE]] [--prepare_data_per_node [PREPARE_DATA_PER_NODE]] [--plugins PLUGINS] [--amp_backend AMP_BACKEND] [--amp_level AMP_LEVEL] [--move_metrics_to_cpu [MOVE_METRICS_TO_CPU]]
            [--multiple_trainloader_mode MULTIPLE_TRAINLOADER_MODE] [--stochastic_weight_avg [STOCHASTIC_WEIGHT_AVG]] [--terminate_on_nan [TERMINATE_ON_NAN]]
        '''
        if base_model == "ldm":
            base="vlpers/third_party/methods/textual_inversion/configs/latent-diffusion/txt2img-1p4B-finetune_save_every_100.yaml"
            actual_resume="checkpoints/textual-inversion/ldm.ckpt"
        elif base_model == "sdm":
            base="vlpers/third_party/methods/textual_inversion/configs/stable-diffusion/v1-finetune_save_every_100.yaml"
            actual_resume="checkpoints/textual-inversion/sdm.ckpt"
        t=""
        gpus="0,"
        max_steps=6100


    class Txt2ImgArgs:
        ddim_eta=0.0
        n_samples=2
        n_iter=2
        scale=7.5
        ddim_steps=50
        if base_model == "ldm":
            ckpt_path="checkpoints/textual-inversion/ldm.ckpt" # THEIR BUG, --ckpt_path is --ckpt for SDM
        elif base_model == "sdm":
            ckpt="checkpoints/textual-inversion/sdm.ckpt"
            config="third_party/textual_inversion/configs/stable-diffusion/v1-inference.yaml"
