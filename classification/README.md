# Classification Factory


## Training
```
! python3 /content/ex/main.py -h
```
```
usage: main.py [-h] [-m ARCH] [-a attention] [--numclasses C] [--epochs N]
               [--start-epoch N] [-b N] [-lr LR] [--momentum M] [--wd W]
               [--resume PATH] [-e] [--hgpath PATH]
               dir

PyTorch ImageNet Training

positional arguments:
  dir                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  -m ARCH, --modelname ARCH
                        model architecture:
  -a attention, --attention attention
                        attention
  --numclasses C        num classes
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batchsize N   batch size
  -lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --wd W, --weight-decay W
                        weight decay (default: 1e-4)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --hgpath PATH         history graph pathre
```


## GradCAm
```
! python3 /content/ex/gradcam.py -h
```
```
usage: gradcam.py [-h] [-m ARCH] [-a ATTENTION] path state_dict_path

GradCAM

positional arguments:
  path                  image path
  state_dict_path       state_dict_path

optional arguments:
  -h, --help            show this help message and exit
  -m ARCH, --modelname ARCH
                        model architecture:
  -a ATTENTION, --attention ATTENTION
                        attention
  
```
