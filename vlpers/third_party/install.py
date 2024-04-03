# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

import subprocess
from rich import print
from pathlib import Path
import shutil

def install_method(model_name, git_url, checkout, patch_file, method_dir):
    print(f"[blue]Downloading {model_name} model...[/blue]")
    subprocess.run(['git', 'clone', git_url, model_name], check=True, cwd=method_dir)
    subprocess.run(['git', 'checkout', checkout], cwd=method_dir / model_name, check=True)
    subprocess.run(['git', 'apply', patch_file], cwd=method_dir / model_name, check=True)
    subprocess.run(['pip', 'install', '-e', '.'], cwd=method_dir / model_name, check=True)
    print(f"[green]{model_name} Installed![/green]")

# Define a dictionary with the model names, git URLs, and patch files
models = {
    'pic2word': {
        'git_url': 'https://github.com/google-research/composed_image_retrieval',
        'checkout': 'main',
        'patch_file': 'pic2word.patch'  # Path from third_party dir
    },
    'PALAVRA': {
        'git_url': 'https://github.com/NVlabs/PALAVRA',
        'checkout': 'main',
        'patch_file': 'palavra.patch'
    },
    'dreambooth': {
        'git_url': 'https://github.com/huggingface/diffusers.git',
        'checkout': '78922ed7c7e66c20aa95159c7b7a6057ba7d590d',
        'patch_file': 'dreambooth.patch'
    },
    'textual_inversion': {
        'git_url': 'https://github.com/rinongal/textual_inversion',
        'checkout': '424192de1518a358f1b648e0e781fdbe3f40c210',
        'patch_file': 'textual_inversion.patch'  
    }
}

def install_third_party():
    '''
    Install the third-party models
    '''
    # Get the current directory
    current_dir = Path(__file__).resolve().parent

    # Create the "methods" subdirectory
    methods_dir = current_dir / 'methods'
    if methods_dir.exists():
        shutil.rmtree(methods_dir)
    methods_dir.mkdir(exist_ok=True)

    patches_dir = current_dir / 'patches'

    # Iterate over the models dictionary and call the install_method function
    for model_name, model_info in models.items():
        try:
            install_method(model_name, model_info['git_url'],model_info['checkout'], patches_dir / model_info['patch_file'], methods_dir)
        except Exception as e:
            print(f"[red]Error installing {model_name} model: {str(e)}[/red]")
            # Delete the method directory
            shutil.rmtree(methods_dir)
            raise e

def remove_third_party():
    '''
    Remove the "methods" subdirectory
    '''
    current_dir = Path(__file__).resolve().parent
    methods_dir = current_dir / 'methods'
    if not methods_dir.exists():
        print("[yellow]Warning: 'methods' directory does not exist![/yellow]")
        return
    shutil.rmtree(methods_dir)
    print("[green]Third-party methods removed successfully![/green]")