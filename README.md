# SetConv: A New Approach for Learning from Imbalanced Data

This repo implements the binary classifier discussed in the paper [SetConv: A New Approach for Learning from Imbalanced Data](https://www.aclweb.org/anthology/2020.emnlp-main.98/), EMNLP, 2020.

## How to Run
### Setup Input Data Dir 

Please go to line 19 at ```SetConv/config/defaults.py``` and set ```_C.Dataset.Dir``` as needed. This directory is the parent directory of your datasets. 

Each dataset is expected to be self-contained in its own directory. 

There are three files in each dataset directory:
* ```datasetName.csv```: a csv file contain the feature vectors extracted from a pre-trained Bert model (one sample per line). The last column provides the label corresponding to each sample.
* ```datasetName_{split_ratio}_train_idx.csv```: a csv file where each line contains an index of a training example.
* ```datasetName_{split_ratio}_valid_idx.csv```: a csv file where each line contains an index of a validation example.
* ```datasetName_{split_ratio}_test_idx.csv```: a csv file where each line contains an index of a test example.

An example of the ```Amazon Books``` dataset is:

```
root/ 
    amz_review/
        books.csv (Please download from https://utdallas.box.com/s/ciw8ruoa94k9avp9d6qojjjdv7zqshi4)
        books_0.6_train_idx.csv
        books_0.6_valid_idx.csv
        books_0.6_test_idx.csv
```

Please remember to add an entry in ```DATASETS``` directory in line 4 at ```SetConv/config/paths_catalog.py```. 

In our example, we should add:
    ```'AmzBooks': 'amz_review/books.csv',```

### Modify the Model Configurations
A default model configuration is provided in ```configs/default_0.6_ours.yaml```. You may modify it as needed.

### Run the Program
Run the program by executing the following command:

```python tools/main.py --config-file configs/default_0.6_ours.yaml```

## Off-The-Shelf Model 
Unfortunately, the IRT dataset contains sensitive tweets collected from users and is not allowed to be redistributed due to privacy issues. If
you need to test our model on that dataset, please contact the dataset owner to obtain the permission. 

Here, we provide 3 models named as ```best_model{1,2,3}.pth```, which are trained on a dummy ```Amazon Books``` review dataset (14407 negative samples and 1477 positive samples uniformly sampled from the original dataset) with the same hyper-parameter configurations. 

```
Train: class 0: 8648, class 1: 882
Valid: class 0: 1438, class 1: 150
Test:  class 0: 4321, class 1: 445
```

The model performance on the test set is:

|Metric|F1|G-Mean|AUC|Spec|Sens|
|-|-|-|-|-|-|
|1|0.78135|0.87544|0.93679|0.90098|0.85062|
|2|0.79873|0.88972|0.95976|0.91736|0.86292|
|3|0.81353|0.89548|0.96117|0.92685|0.86517|
|Avg|0.79787|0.88688|0.95257|0.91506|0.85957|

This dummy dataset is stored in the ```SetConv/AmzBooks/``` folder.

## Citation
If you find this code helpful, please consider to cite:

```
@inproceedings{gao-etal-2020-setconv,
    title = "{S}et{C}onv: {A} {N}ew {A}pproach for {L}earning from {I}mbalanced {D}ata",
    author = "Gao, Yang  and
      Li, Yi-Fan  and
      Lin, Yu  and
      Aggarwal, Charu  and
      Khan, Latifur",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.98",
    doi = "10.18653/v1/2020.emnlp-main.98",
    pages = "1284--1294",
    abstract = "For many real-world classification problems, e.g., sentiment classification, most existing machine learning methods are biased towards the majority class when the Imbalance Ratio (IR) is high. To address this problem, we propose a set convolution (SetConv) operation and an episodic training strategy to extract a single representative for each class, so that classifiers can later be trained on a balanced class distribution. We prove that our proposed algorithm is permutation-invariant despite the order of inputs, and experiments on multiple large-scale benchmark text datasets show the superiority of our proposed framework when compared to other SOTA methods.",
}
```
