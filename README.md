<h1 align="center">
    ConCon-Chi Benchmark
</h1>

![image](https://github.com/hsp-iit/concon-chi_benchmark/assets/47559809/a74f017d-bd2a-4a18-b6b8-e7bc68b8447e)

<h4 align="center">
  ConCon-Chi: Concept-Context Chimera Benchmark for Personalized Vision-Language Tasks
</h4>

<div align="center">
  CVPR 2024
</div>

<div align="center">
  <a href=""><b>Paper (coming soon)</b></a> |
  <a href="https://doi.org/10.48557/QJ1166"><b>Dataset</b></a> |
  <a href=""><b>Website (coming soon)</b></a>
</div>

## Table of Contents

- [Installation](#installation)
- [Reproduce the results](#reproduce-the-paper-results)
- [License](#license)
- [Citing this paper](#citing-this-paper)


## Installation
To install the necessary dependencies we provide a yml file that can be used to create a [conda](https://github.com/conda-forge/miniforge) environment:
```console
conda env create --file environment.yml
conda activate vlpers
```
This will clone and patch some repositories that implement methods compared in the paper. 
For more information about the cloned repositories and details on how to obtain the models' weights read [here](vlpers/third_party/README.md).
Once obtained the weights of the models, set the correct path in the methods' config file in `configs/` (by default **[project_dir]/checkpoints**).

### Datasets
Our dataset ConCon-Chi can be manually downloaded from [here](https://doi.org/10.48557/QJ1166) or by running `download_conconchi` entrypoint inside the conda environment. Read [this](vlpers/datasets/README.md) for a list of datasets supported by the evaluation framework and instructions on how to install them. Datasets should be placed in a directory named `datasets` in the project root.

## Instructions
To evaluate the retrieval performance of a method on a specific dataset run:
```console
python eval_retrieval --method [method name] --dataset [dataset name]
```

This will evaluate the specified method on our dataset using the splits specified in [config/datasets/conconchi.py](configs/datasets/conconchi.py).
Depending on the run configuration this will generate the following files:
- results.csv: tabular file reporting the evaluation metrics for each query
- logs.txt: stdout of the process
- config.json: model and dataset configuration used for the run
- file.patch: every un-committed modification in the repo (for reproducibility)
- concepts.npy & image_pool.npy: embeddings of the learned concepts and of the image_pool (can be used to avoid recomputation in subsequent runs)

## Reproduce the paper results
### Retrieval
#### Table 2
To reproduce the reults of Table 2 you can run:
```console
python eval_retrieval --method [method name] --dataset conconchi
```
changing `[method name]` with one of the following values `[pic2word, palavra, searle, baselines.text_only]` depending on the method you want to replicate.
> [!TIP]
> You can change the type of concept description used by `baselines.text_only` from the configuration file `configs/methods/baselines/text_only.py`
#### Figure 5
Download the dataset you want to plot and run all the cells of `scripts/retrieval_weighted_average.ipynb`.
You can select the dataset to evaluate from the `Config` section of the notebook.

### Generation
#### Table 4
Similarly to retrieval, you can obtain the evaluation metrics for the generation methods by running
```console
python eval_generation.py --method [method name] --dataset conconchi
```
where the available methods are `[dreambooth, textual_inversion, sdm, ldm]`.
This will generate a `results.csv` which can be used by the notebook `scripts/generation_analysis.ipynb` to reproduce the results of Table 4.

## License
The code, except for the content of the [vlpers/third_party/patches](vlpers/third_party/patches) folder, is released under the BSD license (SPDX code: BSD-3-Clause). See [vlpers/third_party/patches/LICENSE](vlpers/third_party/patches/LICENSE) for the licenses of the third party code contained in the patches.

## Citing this paper

```bibtex
@InProceedings{ConConChi_2024_CVPR,
    author    = {Rosasco, Andrea and Berti, Stefano and Pasquale, Giulia and Malafronte, Damiano and Sato, Shogo and Segawa, Hiroyuki and Inada, Tetsugo and Natale, Lorenzo},
    title     = {{ConCon-Chi: Concept-Context Chimera Benchmark for Personalized Vision-Language Tasks}},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
}
```
