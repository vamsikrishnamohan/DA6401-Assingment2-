## Part-A Training a Smaller Network from Scratch
### 1) Q1 to Q3 (Model architecture and wandb sweeps)
Q1toQ3.ipynb loads the training and validation datasets. The different hyperparameter configurations for wandb are specified in the variable sweep_config. 
```python
sweep_config = {
    'method': 'bayes',
    'metric': {
      'name': 'val_acc',
      'goal': 'maximize'   
    },
    'parameters': {
        'conv_filters': {
            'values': [[32, 32, 32, 32, 32], [32, 64, 128, 256, 512]]
        },
        'conv_kernel_sizes': {
            'values': [[3, 3, 3, 3, 3], [5, 3, 3, 3, 3]]
        },
        'conv_activation': {
            'values': ['ReLU', 'SiLU', 'GeLU']
        },
        'dense_neurons': {
            'values': [128, 256] 
        },
        'dense_activation': {
            'values': ['ReLU', 'SiLU']
        },
        'dropout': {
            'values': [0.2, 0.3]
        },
        'use_batchnorm': {
            'values': [True, False]
        },
        'lr': {
            'values': [1e-3, 1e-4]
        },
        'batch_size': {
            'values': [32, 64]
        },
        'use_augmentation': {  
            'values': [True, False]
        }
    }
}
```
The function CNN_train defines the model architecture, trains the model and logs the metrics to wandb. Link for wandb project[Model Runs](https://wandb.ai/da24m026-indian-institute-of-technology-madras/Simple_cnn/reports/DA6401-Assignment-2---VmlldzoxMjE3ODE4OQ#question-2)
### 2) Q4 (Training and evaluating on test set with best model , visualizing test images and filters)
PartA_Q4.ipynb can be used to train a model using the optimal hyperparameters obtained in Q2 and Q3. The code for loading the data, initializing the model architecture and training is similar to the previous notebook. So this notebook can be used to quickly test the code without setting up the wandb sweeps. The optimal hyperparameters are specified as follows. These can be modified for the purpose of testing the code.
```python
best_model = SimpleCNN(
    conv_filters=[32, 64, 128, 256, 512], 
    conv_kernel_sizes=[3, 3, 3, 3, 3],
    conv_activation="GeLU",
    dense_neurons=256,
    dense_activation="SiLU",
    dropout=0.3,
    use_batchnorm=True,
    lr=0.0001
)
```

The code for evaluating this model on the test set (Q4 (a)), visualizing test images (Q4 (b))  is also included in the same notebook.



