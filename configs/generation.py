# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

from datetime import datetime
from pathlib import Path
from clearconf import BaseConfig
from vlpers.utils.misc import git_get_hash

class Config(BaseConfig):
    root = Path('.').resolve()
    class Git:
        hash = git_get_hash()                                                          # hash of the current commit
        make_patch = True                                                              # wether to save a patch of the uncommited changes
        
    class Logging:
        log_dir:Path = Path('logs')                                                     # logs root directory
        exp_dir:str = f'[eval]f"{datetime.now().strftime("%Y%m%d_%H%M%S")}_' \
                      f'{{cfg.Method._name.split(\':\')[1]}}_{{cfg.Data.name}}"'        # experiment directory
        
        save_image_pool:bool = False                                                    # save image pool embeddings
        save_concepts:bool = False                                                      # save concept embeddings                      
        save_images:bool = False                                                        # save generated images