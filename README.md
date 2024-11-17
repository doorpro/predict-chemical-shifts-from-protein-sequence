## PLM-CS  
## Predict protein chemical shifts from sequence


![image](/image/image1.png)
### Train your model
If you want to train your own PLM-CS model, this repository provides all the tools and data. Just follow these steps.


#### Training set
We provide the complete training set data in [RefDB training dataset](./dataset/RefDB_test_remove). Each file in this folder is in nmrstar format, and each file corresponds to a protein. All proteins contained in the *SHIFTX test* are removed from it.
#### Training set processing
For convenience, the reasoning process of the ESM model is separate from the training process of our regression model. Therefore, we first use ESM-650M to process the data. Change the "save_path" in [esm_process.py](./esm_process.py) to your own path. A tensordataset containing the training data will be generated.
#### Train
Modify the path in the [train.py](./train.py) to your own parh. Also, be aware that this can only train a model of one type of atom at a time.
#### Training parameters
Different atom types correspond to different optimizer strategies.You can modify the corresponding parameters in the [train.py](./train.py) according to your trained model. The default number of steps for an iteration is 20,000, but you can change it to 5,000 to achieve very close performance while reducing training time
parameters     | Cα | Cβ | C | Hα | H | N
-------- |--|--|--|--|--|--|
learning rate|0.02|5e-4|0.002|0.01|5e-4|5e-4
optimizer| SGD|Adam|Adam|SGD|Adam|Adam
#### Evaluate

### Use PLM-CS through python SDK
