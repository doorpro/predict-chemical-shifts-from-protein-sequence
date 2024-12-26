## PLM-CS  
## Predict protein chemical shifts from sequence
The github repository address of the project: https://github.com/doorpro/predict-chemical-shifts-from-protein-sequence. 
If you have any comments or questions, email the author: 2260913071@qq.com


![image](https://github.com/doorpro/predict-chemical-shifts-from-protein-sequence/raw/main/image/image1.png "PLM-CS")

## File introduction in the project
[af_pdb](./af_pdb/) deposit the protein structures in the *solution_nmr test set*  
[af_result_csv](./af_result_csv/) deposit the csv files of the protein in *solution_nmr test set* predicted by PLM-CS  
[dataset](./dataset/) deposit all the dataset include the train set(RefDB) and two test set. The processed tensordataset are also included.  
 [plm_cs](./plm_cs/) Inside the folder are installable plm-cs python sdk.  
 [result](./result/) This folder is the default save path of prediction result.  
 [esm_process](./esm_process.py) is a tool we provide for preprocessing sequences.  
 [model](./model.py) holds architecture of the model  
 [requirement](./requirement.txt) listed the relevant dependency libraries and version required for training and testing plm-cs.  
 [setup](./setup.py) if the local installation setup file for plm-cs.  
 [train_your_model](./train_your_model.ipynb) provided the code to completely reproduce the results in our paper.  
 [train](./train.py) provides scripts for training the model if you want to train the model with your own data.
 [utils](./utils.py) provides tools for extract protein sequences and chemical shifts from various types of file.

### Use PLM-CS through python SDK
#### Requirement
    'torch == 2.5.0',
    'torchaudio == 2.5.0',
    'torchvision == 0.20.0',
    'fair-esm == 2.0.0',
    'numpy == 2.1.2',
    'biopython == 1.84',
    'pandas == 2.2.3'

#### Install with pip
```python
pip install plm-cs
```

#### Or install after git clone
After cloning the complete project file locally, run the following command in the folder containing setup.py
```python
pip install .
```

#### Use plm-cs
Using commands similar to the one below, enter the Fasta file or protein sequence and the path to save the result to generate a csv file predicting the chemical shift at the specified location
```python
plm-cs YOURSEQUENCE --result_file your_save_path.csv
plm-cs your_file.fasta --result_file your_save_path.csv
```
Note that the first time you use it, it takes a lot of time because you need to download the weights of the esm model.

You can simply check by: ```plm-cs AAAA```

## Train your model
If you want to train your own PLM-CS model, this repository provides all the tools and data.

### Train with RefDB dataset
If you want to train with the data we provide and get the results in the paper, all the processes are already provided in the ipynb file [train_your_model](./train_your_model.ipynb).

#### Training set
We provide the complete training set data in [RefDB training dataset](./dataset/RefDB_test_remove). Each file in this folder is in nmrstar format, and each file corresponds to a protein. All proteins contained in the *SHIFTX test* are removed from it.

#### Training parameters
Different atom types correspond to different optimizer strategies.You can modify the corresponding parameters in the [train.py](./train.py) according to your trained model. The default number of steps for an iteration is 20,000, but you can change it to 5,000 to achieve very close performance while reducing training time
parameters     | Cα | Cβ | C | Hα | H | N
-------- |--|--|--|--|--|--|
learning rate|0.02|5e-4|0.002|0.01|5e-4|5e-4
optimizer| SGD|Adam|Adam|SGD|Adam|Adam

### Train with your own dataset
#### Training set processing
For convenience, the reasoning process of the ESM model is separate from the training process of our regression model. Therefore, we first use ESM-650M to process the data. In [esm_process.py](./esm_process.py) we provide a transformation function for the esm model, you need to provide three parameters：*protein sequence*, *chemical shifts*, *mask*. The sequence representing the protein, the sequence specifying the chemical shift of the atom, and the mask sequence (if any of the tags for a particular sequence are missing). These three sequences should be of equal length. The function outputs four processed data, you need to concat multiple sequences of data in the batch size dimension and save them as the tensordataset in this manner. 
```python
dataset = TensorDataset(all_esm_vec, all_label, all_mask, all_padding_mask)
```
The final dimension of each parameter should be:
*b&#215;512&#215;1280*, *b&#215;512&#215;1*, *b&#215;512&#215;1*, *b&#215;512&#215;1*

#### Train
Modify the path in the [train.py](./train.py) to your own parh. Also, be aware that this can only train a model of one type of atom at a time.
