# Final Project
## P1 self-supervised learning
```
cd P1
python train_contrast.py
python train_with_pretrained.py
python train_frozen.py
```
running these commands to get a pretrained resnet, a classfier of Cifar10 with pretrained resnet and a baseline classfier with no-pretrained resnet

## P3 NeRF
### Code Base
zipnerf (https://github.com/SuLvXiangXin/zipnerf-pytorch)

### Data Preprocess
put your images in to a folder named "images", experiment photos are in 'data/images'
```
bash scripts/local_colmap_and_resize.sh <path-to-images>
```
after running the colmap, you will find "sparse" folder in your image base directory.


### Train NeRF and Render Video
```
bash scripts/train_llff.sh
```

### Rendered RGB video
https://github.com/519401113/deeplearning-finalPJ/assets/52318421/7da1a5a1-3a92-415a-bb33-62a4edbc709e

