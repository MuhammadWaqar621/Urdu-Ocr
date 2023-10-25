# UTRNet: High-Resolution Urdu Text Recognition



## Using This Repository
### Environment
* Python 3.7
* Pytorch 1.9.1+cu111
* Torchvision 0.10.1+cu111
* CUDA 11.4

### Installation
1. Clone the repository
```
git clone https://github.com/MuhammadWaqar621/Urdu-Ocr.git
```

2. Install the requirements
```
conda create -n urdu_ocr python=3.7
conda activate urdu_ocr
pip3 install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

### Running the code

1. Training
```
python3 train.py --train_data path/to/LMDB/data/folder/train/ --valid_data path/to/LMDB/data/folder/val/ --FeatureExtraction HRNet --SequenceModeling DBiLSTM --Prediction CTC --exp_name UTRNet-Large --num_epochs 100 --batch_size 8

```

2. Testing
```
CUDA_VISIBLE_DEVICES=0 python test.py --eval_data path/to/LMDB/data/folder/test/ --FeatureExtraction HRNet --SequenceModeling DBiLSTM --Prediction CTC --saved_model saved_models/UTRNet-Large/best_norm_ED.pth
```

3. Character-wise Accuracy Testing
* To create character-wise accuracy table in a CSV file, run the following command

```
CUDA_VISIBLE_DEVICES=0 python3 char_test.py --eval_data path/to/LMDB/data/folder/test/ --FeatureExtraction HRNet --SequenceModeling DBiLSTM --Prediction CTC  --saved_model saved_models/UTRNet-Large/best_norm_ED.pth
```

* Visualize the result by running ```char_test_vis```

4. Reading individual images
* To read a single image, run the following command

```
CUDA_VISIBLE_DEVICES=0 python3 read.py --image_path path/to/image.png --FeatureExtraction HRNet --SequenceModeling DBiLSTM --Prediction CTC  --saved_model saved_models/UTRNet-Large/best_norm_ED.pth
```

5. Visualisation of Salency Maps

* To visualize the salency maps for an input image, run the following command

```
python3 vis_salency.py --FeatureExtraction HRNet --SequenceModeling DBiLSTM --Prediction CTC --saved_model saved_models/UTRNet-Large/best_norm_ED.pth --vis_dir vis_feature_maps --image_path path/to/image.pngE
```

### Dataset
1. Create your own lmdb dataset
```
pip3 install fire
python create_lmdb_dataset.py --inputPath data/ --gtFile data/gt.txt --outputPath result/train
```
The structure of data folder as below.
```
data
├── gt.txt
└── test
    ├── word_1.png
    ├── word_2.png
    ├── word_3.png
    └── ...
```
At this time, `gt.txt` should be `{imagepath}\t{label}\n` <br>
For example
```
test/word_1.png label1
test/word_2.png label2
test/word_3.png label3
...
```

# Downloads
## Trained Models
1. [UTRNet-Large](https://csciitd-my.sharepoint.com/:u:/g/personal/ch7190150_iitd_ac_in/EeUZUQsvd3BIsPfqFYvPFcUBnxq9pDl-LZrNryIxtyE6Hw?e=MLccZi)
2. [UTRNet-Small](https://csciitd-my.sharepoint.com/:u:/g/personal/ch7190150_iitd_ac_in/EdjltTzAuvdEu-bjUE65yN0BNgCm2grQKWDjbyF0amBcaw?e=yiHcrA)

## Datasets
1. [UTRSet-Real](https://csciitd-my.sharepoint.com/:u:/g/personal/ch7190150_iitd_ac_in/EXRKCnnmCrpLo8z6aQ5AP7wBN_NujFaPuDPvlOB0Br8KKg?e=eBBuJX)
2. [UTRSet-Synth](https://csciitd-my.sharepoint.com/:u:/g/personal/ch7190150_iitd_ac_in/EUVd7N9q5ZhDqIXrcN_BhMkBKQuc00ivNZ2_jXZArC2f0g?e=Gubr7c)
3. [IIITH (Updated)](https://csciitd-my.sharepoint.com/:u:/g/personal/ch7190150_iitd_ac_in/EXg_48rOkoJBqGpXFav2SfYBMLx18zzgQOtj2kNzpeL4bA?e=ef7lLr) ([Original](https://cvit.iiit.ac.in/research/projects/cvit-projects/iiit-urdu-ocr))
4. [UPTI](https://csciitd-my.sharepoint.com/:u:/g/personal/ch7190150_iitd_ac_in/EVCJZL8PRWVJmRfhXSGdK2ABR17Jo_lW5Ji62JeBBevxcA?e=GgYC8R) ([Source](https://ui.adsabs.harvard.edu/abs/2013SPIE.8658E..0NS/abstract))
5. UrduDoc - Will be made available subject to the execution of a no-cost license agreement. Please contact the authors for the same.

## Text Detection (Supplementary)
The text detection inference code & model based on ContourNet is [here](https://github.com/abdur75648/urdu-text-detection). As mentioned in the paper, it may be integrated with UTRNet for a combined text detection+recognition and hence an end-to-end Urdu OCR.

## Synthetic Data Generation using Urdu-Synth (Supplementary)
The [UTRSet-Synth](https://csciitd-my.sharepoint.com/:u:/g/personal/ch7190150_iitd_ac_in/EUVd7N9q5ZhDqIXrcN_BhMkBKQuc00ivNZ2_jXZArC2f0g?e=Gubr7c) dataset was generated using a custom-designed robust synthetic data generation module - [Urdu Synth](https://github.com/abdur75648/urdu-synth/). 

## End-To-End Urdu OCR Webtool
This was developed by integrating the UTRNet text recognition model with a text detection model for end-to-end Urdu OCR. The webtool may be made available only for non-commercial use upon request. Please contact the authors for the same.

![website](https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition/assets/66300465/511aeffe-d9b3-41aa-8150-ab91f398ae49)



## Contact
* [Muhammad Waqar](https://www.linkedin.com/in/muhammad-waqar-1a594411a/)
* [Email](waqarsahi621@gmail.com)


