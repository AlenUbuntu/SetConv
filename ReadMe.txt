This repo implements the binary classifier discussed in the paper "SetConv: A New Approach for Learning from Imbalanced Data", EMNLP, 2020.

## How to Run
1. Setup Input Data Dir 
Please go to line 19 at  SetConv/config/defaults.py and set _C.Dataset.Dir as needed. This directory is the parent directory of your datasets. Each dataset
is expected to be self-contained in its own directory. There are three files in the dataset directory:
* datasetName.csv: a csv file contain the feature vectors extracted from a pre-trained Bert model (one sample per line). The last column contains the labels corresponding to each sample.
* datasetName_{split_ratio}_train_idx.csv: a csv file where each line contains an index of a training example.
* datasetName_{split_ratio}_test_idx.csv: a csv file where each line contains an index of a test example.

An example of the Amazon Books dataset is:

root/ 
    amz_review/
        books.csv
        books_0.6_train_idx.csv
        books_0.6_test_idx.csv

Please remember to add an entry in `DATASETS` directory in line 4 at SetConv/config/paths_catalog.py. In this case, we should add:
    'AmzBooks': 'amz_review/books.csv',

2. modify configs/default_0.6_ours.yaml as needed.

3. run the program by the following command:

python tools/main.py --config-file configs/default_0.6_ours.yaml

## Off-The-Shelf Model 
Unfortunately, the IRT dataset contains sensitive tweets collected from users and is not allowed to be redistributed due to privacy issues. If
you need to test our model on that dataset, please contact the dataset owner to obtain the permission. 

Thus, we provide a model trained on a dummy public available Amazon Books review dataset (14407 negative samples, 1477 positive samples), which is named as best_modeli.pth. 
Train: class 0: 8648, class 1: 882
Valid: class 0: 1438, class 1: 150
Test:  class 0: 4321, class 1: 445

Here are the model performance on the test set for 3 independent runs:

F1: 0.78135 | G-Mean:0.87544 | AUC: 0.93679 | Spec: 0.90098 | Sens: 0.85062
F1: 0.79873 | G-Mean:0.88972 | AUC: 0.95976 | Spec: 0.91736 | Sens: 0.86292
F1: 0.81353 | G-Mean:0.89548 | AUC: 0.96117 | Spec: 0.92685 | Sens: 0.86517