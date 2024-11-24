# SalientSleepNet
Code for the model in the paper SalientSleepNet: Multimodal Salient Wave Detection Network for Sleep Staging (Accepted by IJCAI 2021).
![overall structure](figure/u2structure.png)

## Datasets
>We evaluate the performance of SalientSleepNet on Sleep-EDF-39 and Sleep-EDF-153 datasets, which are open-access databases of polysomnography (PSG) recordings.

## Requirements
* Python 3.7
* Tensorflow 2.3.1
* PyYAML 5.1.1
* Scikit-learn 0.21.2
* Numpy 1.16.4
* Matplotlib 3.1.0
* mne 2.10

## Function of file
* `prepare_npz`
  * Prepare npz files from raw data.
* `load_files.py`
  * Provide functions to load sleep signal files (npz files).
* `preprocess.py`
  * Provide functions to preprocess sleep signal sequence (E.g., regularization, divide groups).
* `evaluation.py`
  * Provide evaluation functions for summary.py to call.
* `loss_function.py` 
  * Generate loss function.
* `construct_model.py`
  * Generate submodule of model (E.g., u-unit).
* `model.py`
  * Generate single-branch and two-branch model.
* `train.py` 
  * Preprocess data and train model.
* `summary.py` 
  * Call functions in evaluation.py to evaluate training results and Plot result images.
* `hyperparameters.yaml`
  * Provide hyperparameters for training.

## Examples of use
### 1.Get data
You can get Sleep-EDF-39 & Sleep-EDF-153 with:
>$ wget https://www.physionet.org/static/published-projects/sleep-edfx/sleep-edf-database-expanded-1.0.0.zip

### 2.Data preparation
>$ python prepare_npz/prepare_npz_data -d ./sleep-edf-database-expanded/sleep-cassette -o ./data/sleep-edf-153/npzs
* `--data_dir -d` File path to the edf file that contain sleeping info.
* `--output_dir -o` Directory where to save outputs.
* `--select_ch -s` Choose the channels for training.

### 3.Training
>$ python ./train.py --gpus 1 -d "./data/sleep-edf-153/npzs"

* `--gpus -g` The number of GPU to use (max: 4).
* `--modal -m` The number of  modal to train. Set this to 0 for single modal, 1 for  multiple modal.
* `--data_dir -d` The address of data (directory of npz files).
* `--output_dir -o` The address of output.
* `--valid -v` The total fold number of k-fold validation (E.g., 20 means use 20-fold validation).
* `--from_fold` The starting fold number of  training in this program (E.g., 4 means start training from the 4-th fold).
* `--train_fold` The training folds of this program（E.g., Set from_fold to 4 and set train_fold to 7 to make program training 4,5,6,7 folds. You need start 5 program to make them training 0-3, 4-7, 8-11, 12-15, 16-19 folds separately to implement 20-fold validation).

### 4.Evaluation
>$ python ./summary.py -d "./data/sleep-edf-153/npzs"

* `--modal -m` The number of  modal to train. Set this to 0 for single modal, 1 for  multiple modal.
* `--data_dir -d` The address of data (directory of npz files).
* `--output_dir -o` The address of output.
* `--valid -v` The total fold number of k-fold validation (E.g., 20 means use 20-fold validation).
