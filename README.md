# LET-NET Training Code 

## Requirements

- numpy~=1.24.4
- opencv-python~=4.9.0.80
- torch~=2.2.2
- numba~=0.58.1
- pillow~=10.3.0
- torchvision~=0.17.2
- matplotlib~=3.7.5
- tqdm~=4.66.2
- h5py~=3.10.0
- tensorboard~=2.14.0

## Datasets

- HPatches (https://github.com/hpatches/hpatches-dataset)
- megadepth (https://www.cs.cornell.edu/projects/megadepth/)

Download the datasets and change path value in `main.py(line 75-77)`

If you cannot download the megadepth dataset, you can use this link to download the dataset.

- [megadepth](https://pan.baidu.com/s/1D61PeZ6tQovehvyawGUpcA)(pass: ypnw)

## Training

run:

``` python
python main.py
```

tensorboard visualization

```shell
tensorboard --logdir=./log_main/ --port=6006
```


## Thanks

ALIKE training code https://github.com/Shiaoming/ALIKE

