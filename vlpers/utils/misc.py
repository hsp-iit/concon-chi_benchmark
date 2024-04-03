# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

from datetime import datetime
import json
from pathlib import Path
from vlpers.utils.logging import logger
from git import Repo


def set_reproducibility():
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    cudnn.deterministic = True

 
def save_config(path, config):
    # Save config
    path = Path(path)
    if (path / 'config.json').exists():
        logger.warning(f"Configuration already logged for a previous run: {path}")
        return
    
    with (path / 'config.json').open('w') as file:
        json.dump(config.to_dict(), file, indent=2)
        
def save_results(path, results, filename='results.csv'): 
    # Save results
    path = Path(path)
    if (path / 'results.csv').exists():
        logger.warning(f"Results already logged for a previous run: {path}")
        return
    
    results.to_csv((path / filename).as_posix(), sep=';', index=False)
    
def save_images(paths, images):
    # Save multiple images in the specified directories
    
    for path, image_samples in zip(paths, images):
    
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        for i, img in enumerate(image_samples):
            img.save(path / f'img_{i}.jpg')
    
    
def git_get_hash():
    repo = Repo(search_parent_directories=True)
    hash = repo.head.object.hexsha[:7]

    if repo.is_dirty() or len(repo.untracked_files) != 0:
        hash = f'{hash}_modified'
    
    return hash

def git_check_workspace(allow_uncommitted:bool=False, allow_untracked:bool=False, make_patch:bool=None, path:str=None):
    ''' 
        Assuming that there are uncommitted and untracked changes:
        - If make_patch=False and allow_uncommitted/untrucked=False the run is aborted
        - If make_patch=True a patch file containing the changes will
          be created in path.
        - In all the other cases the function just returns
    '''
    path = Path(path)
    repo = Repo(search_parent_directories=True)
    untracked_files = repo.untracked_files
    
    if uncommitted := repo.is_dirty():
        logger.warning(f"Unstaged/uncommited changes detected")
        
    if untracked := len(untracked_files) != 0:
        logger.warning(f"Untracked files detected")
        
    if not (uncommitted or untracked):
        return
    
    if (path / 'file.patch').exists():
        logger.warning(f"Patch already logged for a previous run: {path}")
        return

    if make_patch is None:
        print('The following file will be added to the patch:')
        print(repo.git.execute(['git','diff', '--name-only', 'HEAD']))
        print(*untracked_files, sep='\n')
        make_patch = input('Continue? [y]/n: ') in ['y', '']
    
    if not make_patch:
        if (uncommitted and not allow_uncommitted) or \
           (untracked and not allow_untracked):
            logger.error('Aborting')
            exit(1) 
        return
   
    if (path is None) or not Path(path).exists():
        raise ValueError('Specify a valid directory for the patch')
    
    # Untrucked files need to be added to the index to appear in the patch
    if untracked:
        repo.index.add(untracked_files)
    
    repo.git.execute(['git','diff', f'--output={path.as_posix()}/file.patch', 'HEAD'])
    
    # After the patch is done we unstage the file we added to the index
    if untracked:
        repo.git.execute(['git', 'restore', '--staged', *untracked_files])