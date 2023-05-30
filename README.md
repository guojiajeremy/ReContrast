# ReContrast

Official PyTorch Implementation of
"ReContrast: Domain-Specific Anomaly Detection via Contrastive Reconstruction".
Under Review.

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