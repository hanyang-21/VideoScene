<p align="center">
  <h1 align="center">VideoScene: Distilling Video Diffusion Model to Generate 3D Scenes in One Step</h1>
  <p align="center">
    <a href="https://hanyang-21.github.io/">Hanyang Wang</a><sup>*</sup>,
    <a href="https://liuff19.github.io/">Fangfu Liu</a><sup>*</sup>,
    <a href="https://github.com/hanyang-21/VideoScene">Jiawei Chi</a>,
    <a href="https://duanyueqi.github.io/">Yueqi Duan</a>
    <br>
    <sup>*</sup>Equal Contribution.
    <br>
    Tsinghua University
  </p>
  <h3 align="center">CVPR 2025 Hightlight 🔥</h3>
  <h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2403.20309-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2504.01956) 
[![Home Page](https://img.shields.io/badge/Project-Website-green.svg)](https://hanyang-21.github.io/VideoScene)
</h5>
  <!-- <h3 align="center"><a href="https://arxiv.org/abs/">Paper</a> | <a href="">Project Page</a> | <a href="">Pretrained Models</a> </h3> -->
<!--   <div align="center">
    <a href="https://news.ycombinator.com/item?id=41222655">
      <img
        alt="Featured on Hacker News"
        src="https://hackerbadge.vercel.app/api?id=41222655&type=dark"
      />
    </a>
  </div> -->

</p>

<div align="center">
VideoScene is a one-step video diffusion model that bridges the gap from video to 3D.
</div>
</br>


https://github.com/user-attachments/assets/dca733b1-b78f-49ac-ae47-5d1b1e8a689b

Building on [ReconX](https://github.com/liuff19/ReconX), VideoScene has achieved a turbo-version advancement.



## Installation

To get started, clone this project, create a conda virtual environment using Python 3.10+, and install the requirements:

1. Clone VideoScene.
```bash
git clone https://github.com/hanyang-21/VideoScene
cd VideoScene
```

2. Create the environment, here we show an example using conda.
```bash
conda create -y -n videoscene python=3.10
conda activate videoscene
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

3. Optional, compile the cuda kernels for RoPE (as in CroCo v2).
```bash
# NoPoSplat relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd src/model/encoder/backbone/croco/curope/
python setup.py build_ext --inplace
cd ../../../../../..
```

## Acquiring Datasets

### RealEstate10K and ACID

Our VideoScene uses the same training datasets as pixelSplat. Below we quote pixelSplat's [detailed instructions](https://github.com/dcharatan/pixelsplat?tab=readme-ov-file#acquiring-datasets) on getting datasets.

> pixelSplat was trained using versions of the RealEstate10k and ACID datasets that were split into ~100 MB chunks for use on server cluster file systems. Small subsets of the Real Estate 10k and ACID datasets in this format can be found [here](https://drive.google.com/drive/folders/1joiezNCyQK2BvWMnfwHJpm2V77c7iYGe?usp=sharing). To use them, simply unzip them into a newly created `datasets` folder in the project root directory.

> If you would like to convert downloaded versions of the Real Estate 10k and ACID datasets to our format, you can use the [scripts here](https://github.com/dcharatan/real_estate_10k_tools). Reach out to us (pixelSplat) if you want the full versions of our processed datasets, which are about 500 GB and 160 GB for Real Estate 10k and ACID respectively.

## Downloading checkpoints

* download our [pretrained weights](https://drive.google.com/drive/folders/1FB5Rpr4uEVo9U2BBXai1Fc57kHDpp5yj), and save them to `checkpoints`.

* for customized image inputs, get the NoPoSplat [pretrained models](https://huggingface.co/botaoye/NoPoSplat/resolve/main/mixRe10kDl3dv_512x512.ckpt), and save them to `checkpoints/noposplat`.


* for RealEstate10K datasets, get the MVSplat [pretrained models](https://drive.google.com/drive/folders/14_E_5R6ojOWnLSrSVLVEMHnTiKsfddjU), and save them to `checkpoints/mvsplat`.

## Running the Code

### Gradio Demo
In this demo, you can run VideoScene on your machine to generate a video with unposed input views.

* select image pairs that depicts the same scene and hit "RUN" for a video of the scene.

```bash
python -m noposplat.src.app \
    checkpointing.load=checkpoints/noposplat/mixRe10kDl3dv_512x512.ckpt \
    test.video=checkpoints/model.safetensors

# also "bash demo.sh"
```
* the generated video will be stored under `outputs/gradio`

### Inference

To generate videos on RealEstate10K dataseets, we use a [MVSplat](https://github.com/donydchen/mvsplat) pretrained model,

* run the following:

```bash
# re10k
python -m mvsplat.src.main +experiment=re10k \
checkpointing.load=checkpoints/mvsplat/re10k.ckpt \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=mvsplat/assets/evaluation_index_re10k_video.json \
test.save_video=true \
test.save_image=false \
test.compute_scores=false \
test.video=checkpoints/model.safetensors

# also "bash inference.sh"
```

* the generated video will be stored under `outputs/test`


## BibTeX

```bibtex
@misc{wang2025videoscenedistillingvideodiffusion,
      title={VideoScene: Distilling Video Diffusion Model to Generate 3D Scenes in One Step}, 
      author={Hanyang Wang and Fangfu Liu and Jiawei Chi and Yueqi Duan},
      year={2025},
      eprint={2504.01956},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.01956}, 
}
```

## Acknowledgements

This project is developed with several fantastic repos: [ReconX](https://github.com/liuff19/ReconX), [MVSplat](https://github.com/donydchen/mvsplat), [NoPoSplat](https://github.com/cvg/NoPoSplat), [CogVideo](https://github.com/THUDM/CogVideo), and [CogvideX-Interpolation](https://github.com/feizc/CogvideX-Interpolation). Many thanks to these projects for their excellent contributions!
