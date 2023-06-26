# Final Project (NeRF Project)

## Code Base
zipnerf (https://github.com/SuLvXiangXin/zipnerf-pytorch)

## Data Preprocess
put your images in to a folder named "images"
```
bash scripts/local_colmap_and_resize.sh <path-to-images>
```
after running the colmap, you will find "sparse" folder in your image base directory.


## Train NeRF and Render Video
```
bash scripts/train_llff.sh
```

