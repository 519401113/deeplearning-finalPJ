# Final Project
## P1 self-supervised learning
```
cd P1
python train_contrast.py
python train_with_pretrained.py
python train_frozen.py
```
Running these commands to get a pretrained resnet, a classfier of Cifar10 with pretrained resnet and a baseline classfier with no-pretrained resnet. 
### result
We pretrain the resnet encoder on Cifar10 training set. 
30.1% acc with no-pretrained resnet and 41.3% acc with pretrained resnet

## P2 vision transformer
```
cd P2
python main.py
```
Running these commands to train a vision transformer with parameters 10756452, which is closed to ResNet-18 with parameters 11220132.
### result
ViT gets a 43.52% accuracy on CIFAR100 test dataset.

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

