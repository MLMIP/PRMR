# Pseudo-label Refinement and Model-State Recovery for Continual Test-Time Adaptation on Semantic Segmentation

## Environment setup
```
conda create -n prmr python=3.8 -y
conda activate prmr
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html
pip install -r requirements.txt
```

## Prepare Dataset
ACDC could directly download it from [ACDC](https://acdc.vision.ee.ethz.ch/#about). 
Please set the acdc data_root in 
```
local_configs/_base_/datasets/acdc_1024x1024_repeat.py
```

## Prepare checkpoint
We use the pretrained SegFormer-B5 from here [Segformer](https://github.com/MLMIP/PRMR/releases/tag/v1.0.0/segformer.b5.1024x1024.city.160k.pth), and save in ./ckpt.

The diffusion model used in [here](https://github.com/MLMIP/PRMR/releases/tag/v1.0.0/diffusion_model.pt), and save in ./ckpt.
## Run
```
python3 tools/prmr.py
```


## Acknowledgement

Our code is based on CoTTA, SVDP and CCDM. Thanks them for releasing their codes.

