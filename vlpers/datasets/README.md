Here is a list of the supported datasets along with a link to the instructions on how to install them.

| Method | Instructions | Additional Notes |
| --- | --- | --- |
| CIRR | Instructions [here](https://github.com/Cuberick-Orion/CIRR) | |
| CIRCO | Instructions [here](https://github.com/miccunifi/CIRCO) | |
| FashionIQ | Instructions [here](https://github.com/XiaoxiaoGuo/fashion-iq) | For the annotations clone [this](https://github.com/XiaoxiaoGuo/fashion-iq) repository in `datasets/fashion-iq`, for the data download the images from [this](https://github.com/hongwang600/fashion-iq-metadata) list of urls in `datasets/fashion-iq/images`. Some images may be  missing. The dataloader manages missing images, but the results can differ w.r.t. the ones reported in the paper.|
| PerVL-DeepFashion2 | Instructions [here](https://github.com/NVlabs/PerVLBenchmark?tab=readme-ov-file#pervl-deepfashion2) | |

> [!IMPORTANT]
> By default the wrappers for the datasets will look for the data inside of the folder `datasets` in the project directory. If you wanna change the datasets location change the path inside of `configs/datasets/[dataset]`.
