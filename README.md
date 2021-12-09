# Suprisingly Semi-Supervised Domain Adaptation
In this project we use two step learning process which is defined as follows, 
1. Pretraining the base network and porting it to some other place
2. Training a PAC based model for domain adaptation.

The entire project has following components which are as follows:
* _Preprocessing pipeline_ : To preprocess the dataset for rotation training and train test split
* _Dataloader pipeline_ : Component to load the dataset into the models
* _Rotation training_ : A script to train rotationent for pretraining ops
* _Supcon training_ : A script to train supervised contrastive learning methods for pretraining ops
* _PAC training_ : A script to train the PAC model for domain adaptation methods

## Preprocessing Pipeline
This pipeline is to preprocess the dataset for the pre training and dataset

### Rotation Split
This script is used for creating a dataset for the rotation pre training.

The ussage for the script could be defined by passing the _-h_ or _--help_ flag.
```python
python3 preprocessing/rotation_dataset.py --help
usage: rotation_dataset.py [-h] --input_path INPUT_PATH --output_path OUTPUT_PATH [--image_size RESIZE_SHAPE]

Script to create a rotation dataset for training rotation net

optional arguments:
  -h, --help            show this help message and exit
  --input_path INPUT_PATH
                        Input path for loading the images from input
  --output_path OUTPUT_PATH
                        Output path for saving the images from input
  --image_size RESIZE_SHAPE
                        resize image shape
```

### Train Test Split
To perfrom the train test and validation split for the dataset we can use the train test split in the datset which could be used in following way

```python
python3 preprocessing/train_test_split.py --help
usage: train_test_split.py [-h] --input_path INPUT_PATH --output_path OUTPUT_PATH [--test_ratio TEST_RATIO] [--split_type {un,semi,semi_variation}]

Script to create a train test split dataset for the multiple domain based unsupervised testing

optional arguments:
  -h, --help            show this help message and exit
  --input_path INPUT_PATH
                        Input path for loading the images from input
  --output_path OUTPUT_PATH
                        Output path for saving the images from input
  --test_ratio TEST_RATIO
                        Describe test ratio in which data needs to be split
  --split_type {un,semi,semi_variation}
                        Describe what kind of split is required for the dataset
```

As it could be seen the train test split could be carried out for all the three methods we just need to pass the flag for the split type.

## Pretraining Scripts
These are the scripts to perform the pre training on the feature extractot models i.e. typically resnet50 model.

### Train rotnet
Script to train the rotnet pretraining.
```python
python3 train_rotnet.py --help
usage: train_rotnet.py [-h] [--epoch EPOCH] [--batch_size BATCH_SIZE] [--lr LR] --path_to_train_dir PATH_TO_TRAIN_DIR --path_to_test_dir PATH_TO_TEST_DIR
                       --path_to_checkpoint PATH_TO_CHECKPOINT --path_to_save_weights PATH_TO_SAVE_WEIGHTS --path_to_tb PATH_TO_TB [--height HEIGHT]
                       [--width WIDTH] [--channel CHANNEL]

Script to train the models for the domain adaptation

optional arguments:
  -h, --help            show this help message and exit
  --epoch EPOCH         number of total epoches
  --batch_size BATCH_SIZE
                        number of samples in one batch
  --lr LR               initial learning rate for adam
  --path_to_train_dir PATH_TO_TRAIN_DIR
                        path to dataset directory
  --path_to_test_dir PATH_TO_TEST_DIR
                        path to dataset directory
  --path_to_checkpoint PATH_TO_CHECKPOINT
                        path to directory where checkpoints needs to be saved
  --path_to_save_weights PATH_TO_SAVE_WEIGHTS
                        path to directory where checkpoints needs to be saved
  --path_to_tb PATH_TO_TB
                        file name to save the model performance as tensorboard log files
  --height HEIGHT       height of input images, default value is 128
  --width WIDTH         width of input images, default value is 128
  --channel CHANNEL     channel of input images, default value is 3
```

### Train Supervised Contrastive Learning
Script to train the sueprvised contrastive learning pretraining.
```python
python3 train_supcon.py --help
usage: train_supcon.py [-h] [--epoch EPOCH] [--batch_size BATCH_SIZE] [--lr LR] --path_to_data_dir PATH_TO_DATA_DIR --path_to_save_weights
                       PATH_TO_SAVE_WEIGHTS --log_file_path LOG_FILE_PATH [--height HEIGHT] [--width WIDTH] [--channel CHANNEL]

Script to train the models for the domain adaptation

optional arguments:
  -h, --help            show this help message and exit
  --epoch EPOCH         number of total epoches
  --batch_size BATCH_SIZE
                        number of samples in one batch
  --lr LR               initial learning rate for adam
  --path_to_data_dir PATH_TO_DATA_DIR
                        path to dataset directory
  --path_to_save_weights PATH_TO_SAVE_WEIGHTS
                        path to directory where checkpoints needs to be saved
  --log_file_path LOG_FILE_PATH
                        file name to save the model training loss in csv file
  --height HEIGHT       height of input images, default value is 128
  --width WIDTH         width of input images, default value is 128
  --channel CHANNEL     channel of input images, default value is 3
```

## Training PAC
Training a pac could be done in two fashion i.e., supervised setting and unsupervised setting which could be used in following way.

```python
python3 main.py --help
usage: main.py [-h] {semi,unsupervised,eval} ...

Script to train the models for the domain adaptation

positional arguments:
  {semi,unsupervised,eval}
                        semi, unsupervised
    semi                train the classification model in semi supervised fashion
    unsupervised        train the classification model in semi supervised fashion
    eval                Evaluate the performance for the target domain

optional arguments:
  -h, --help
```

### Semi Supervised Setting
This script could be used in vanilla Semi Supervised and semi supervised variation setting.
```python
python3 main.py semi --help

usage: main.py semi [-h] [--epoch EPOCH] [--batch_size BATCH_SIZE] [--lr LR] --path_to_source_dir PATH_TO_SOURCE_DIR --path_to_target_dir
                    PATH_TO_TARGET_DIR --path_to_unlabeled_dir PATH_TO_UNLABELED_DIR [--path_to_pretrained_weights PATH_TO_PRETRAINED_WEIGHTS]
                    --path_to_save_weights PATH_TO_SAVE_WEIGHTS --log_file_path LOG_FILE_PATH [--height HEIGHT] [--width WIDTH] [--channel CHANNEL]

optional arguments:
  -h, --help            show this help message and exit
  --epoch EPOCH         number of total epoches
  --batch_size BATCH_SIZE
                        number of samples in one batch
  --lr LR               initial learning rate for adam
  --path_to_source_dir PATH_TO_SOURCE_DIR
                        path to source dataset directory
  --path_to_target_dir PATH_TO_TARGET_DIR
                        path to target dataset directory
  --path_to_unlabeled_dir PATH_TO_UNLABELED_DIR
                        path to unlabeled dataset directory
  --path_to_pretrained_weights PATH_TO_PRETRAINED_WEIGHTS
                        path to directory from where pretrained weights needs to be loaded, default to rotnet
  --path_to_save_weights PATH_TO_SAVE_WEIGHTS
                        path to directory where checkpoints needs to be saved
  --log_file_path LOG_FILE_PATH
                        file name to save the model training loss in csv file
  --height HEIGHT       height of input images, default value is 128
  --width WIDTH         width of input images, default value is 128
  --channel CHANNEL     channel of input images, default value is 3
```

### Unsupervised Setting
The script is used to train the model in unsupervised setting.
```python
python3 main.py unsupervised --help

usage: main.py unsupervised [-h] [--epoch EPOCH] [--batch_size BATCH_SIZE] [--lr LR] --path_to_source_dir PATH_TO_SOURCE_DIR --path_to_unlabeled_dir
                            PATH_TO_UNLABELED_DIR [--path_to_pretrained_weights PATH_TO_PRETRAINED_WEIGHTS] --path_to_save_weights PATH_TO_SAVE_WEIGHTS
                            --log_file_path LOG_FILE_PATH [--height HEIGHT] [--width WIDTH] [--channel CHANNEL]

optional arguments:
  -h, --help            show this help message and exit
  --epoch EPOCH         number of total epoches
  --batch_size BATCH_SIZE
                        number of samples in one batch
  --lr LR               initial learning rate for adam
  --path_to_source_dir PATH_TO_SOURCE_DIR
                        path to source dataset directory
  --path_to_unlabeled_dir PATH_TO_UNLABELED_DIR
                        path to unlabeled dataset directory
  --path_to_pretrained_weights PATH_TO_PRETRAINED_WEIGHTS
                        path to directory from where pretrained weights needs to be loaded, default to rotnet
  --path_to_save_weights PATH_TO_SAVE_WEIGHTS
                        path to directory where checkpoints needs to be saved
  --log_file_path LOG_FILE_PATH
                        file name to save the model training loss in csv file
  --height HEIGHT       height of input images, default value is 128
  --width WIDTH         width of input images, default value is 128
  --channel CHANNEL     channel of input images, default value is 3
```
### Eval Script
The script to evaluate the performance of the trained model on the validation set
```python
python3 main.py eval --help

usage: main.py eval [-h] [--batch_size BATCH_SIZE] --path_to_domain_dir PATH_TO_DOMAIN_DIR --path_to_saved_weights PATH_TO_SAVED_WEIGHTS [--height HEIGHT]
                    [--width WIDTH] [--channel CHANNEL]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        number of samples in one batch
  --path_to_domain_dir PATH_TO_DOMAIN_DIR
                        path to domain directory to eval
  --path_to_saved_weights PATH_TO_SAVED_WEIGHTS
                        path to directory from where pretrained weights needs to be loaded, default to rotnet
  --height HEIGHT       height of input images, default value is 128
  --width WIDTH         width of input images, default value is 128
  --channel CHANNEL     channel of input images, default value is 3
```

## Dataset
The Office-Home dataset is used in running all the experiments. Which could be downloaded from [following link](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw).