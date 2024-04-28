# ReContrast

Official PyTorch Implementation of
"ReContrast: Domain-Specific Anomaly Detection via Contrastive Reconstruction".

NeurIPS 2023. [paper](https://arxiv.org/abs/2306.02602) [proceddings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/228b9279ecf9bbafe582406850c57115-Abstract-Conference.html)

```
@inproceedings{NEURIPS2023_228b9279,
 author = {Guo, Jia and lu, shuai and Jia, Lize and Zhang, Weihang and Li, Huiqi},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Oh and T. Neumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
 pages = {10721--10740},
 publisher = {Curran Associates, Inc.},
 title = {ReContrast: Domain-Specific Anomaly Detection via Contrastive Reconstruction},
 url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/228b9279ecf9bbafe582406850c57115-Paper-Conference.pdf},
 volume = {36},
 year = {2023}
}
```
## 1. Environments

Create a new conda environment and install required packages.

```
conda create -n my_env python=3.8.12
conda activate my_env
pip install -r requirements.txt
```
Experiments are conducted on NVIDIA GeForce RTX 3090 (24GB). Same GPU and package version are recommended. 

## 2. Prepare Datasets
Noted that `../` is the upper directory of ReContrastAD. It is where we keep all the datasets by default.
You can also alter it according to your need, just remember to modify the `train_path` and `test_path` in the code. 

### MVTec AD

Download the MVTec-AD dataset from [URL](https://www.mvtec.com/company/research/datasets/mvtec-ad).
Unzip the file to `../mvtec_anomaly_detection/`.
```
|-- mvtec_anomaly_detection
    |-- bottle
    |-- cable
    |-- capsule
    |-- ....
```


### VisA

Download the VisA dataset from [URL](https://github.com/amazon-science/spot-diff).
Unzip the file to `../VisA/`. Preprocess the dataset to `../VisA_pytorch/` in 1-class mode by their official splitting 
[code](https://github.com/amazon-science/spot-diff).

You can also run the following command for preprocess, which is the same to their official code.

```
python ./prepare_data/prepare_visa.py --split-type 1cls --data-folder ../VisA --save-folder ../VisA_pytorch --split-file ./prepare_data/split_csv/1cls.csv
```
`../VisA_pytorch/` will be like:
```
|-- VisA_pytorch
    |-- 1cls
        |-- candle
            |-- ground_truth
            |-- test
                    |-- good
                    |-- bad
            |-- train
                    |-- good
        |-- capsules
        |-- ....
```
 
### OCT2017
Creat a new directory `../OCT2017`. Download ZhangLabData form [URL](https://data.mendeley.com/datasets/rscbjbr9sj/3).
Unzip the file, and move everything in `ZhangLabData/CellData/OCT` to `../OCT2017/`. The directory should be like:
```
|-- OCT2017
    |-- test
        |-- CNV
        |-- DME
        |-- DRUSEN
        |-- NORMAL
    |-- train
        |-- CNV
        |-- DME
        |-- DRUSEN
        |-- NORMAL
```

### APTOS
Creat a new directory `../APTOS`.
Download APTOS 2019 form [URL](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data).
Unzip the file to `../APTOS/original/`. Now, the directory would be like:
```
|-- APTOS
    |-- original
        |-- test_images
        |-- train_images
        |-- test.csv
        |-- train.csv
```
Run the following command to preprocess the data to `../APTOS/`.
```
python ./prepare_data/prepare_aptos.py --data-folder ../APTOS/original --save-folder ../APTOS
```
The directory would be like:
```
|-- APTOS
    |-- test
        |-- NORMAL
        |-- ABNORMAL
    |-- train
        |-- NORMAL
    |-- original
```
You can delete `original` if you want.

### ISIC2018
Creat a new directory `../ISIC2018`.
Go to the ISIC 2018 official [website](https://challenge.isic-archive.com/data/#2018).
Download "Training Data","Training Ground Truth", "Validation Data", and "Validation Ground Truth" of Task 3.
Unzip them to `../ISIC2018/original/`. Now, the directory would be like:
```
|-- ISIC2018
    |-- original
        |-- ISIC2018_Task3_Training_GroundTruth
        |-- ISIC2018_Task3_Training_Input
        |-- ISIC2018_Task3_Validation_GroundTruth
        |-- ISIC2018_Task3_Validation_Input
```
Run the following command to preprocess the data to `../ISIC2018/`.
```
python ./prepare_data/prepare_isic2018.py --data-folder ../ISIC2018/original --save-folder ../ISIC2018
```
The directory would be like:
```
|-- ISIC2018
    |-- test
        |-- NORMAL
        |-- ABNORMAL
    |-- train
        |-- NORMAL
    |-- original
```
You can delete `original` if you want.


## 3. Run Experiments
MVTec AD
```
python recontrast_mvtec.py
```
If you want to specify a GPU
```
python recontrast_mvtec.py --gpu 0
```

VisA
```
python recontrast_visa.py
```

APTOS
```
python recontrast_aptos.py
```

OCT2017
```
python recontrast_oct.py
```

ISIC2018
```
python recontrast_isic.py
```

### Model-Unifed Multi-Class Setting
Following the setting proposed by UniAD, we train an unifed model for all classes of each dataset (15 classes for MVTec AD, 12 classes for VIsA).

```
python recontrast_mvtec_multiclass.py
```
```
python recontrast_visa_multiclass.py
```

### Stable Training

Our method (as well as many other UAD methods) suffers from some extent of training instability due to optimizer and batchnorm (BN) related issue,
as discussed in Appendix E. By default, the BN layers of encoder
are set to train mode during training. Because training instability and performance drop are observed
for some categories, the BN of encoder is set to eval mode for such categories. (the choice of encoder BN mode
and training loss spikes can be easily addressed by a validation set, which however is not allowed in UAD).

We explore some tricks that enable training with more stability when set encoder BN to train mode for all categories, 
which produces comparably good performances.
1. In encoder, we use pre-trained running_var if the batch variance of a BN channel is lower than min(5e-4, running_var)
2. We reset the decoder Adam optimizer every 500 iterations to clear historical first-order and second-order gradient.

```
python recontrast_mvtec_stable.py
```
```
python recontrast_visa_stable.py
```

### Acknowledgement
Many thanks to [RD4AD](https://github.com/hq-deng/RD4AD), for their easy-to-read code base.
