## Part-B Using Pre-trained Models for Image Classification

### 1. Dataset   

The code: [iNaturalist dataset](https://github.com/vamsikrishnamohan/DA6401-Assingment2-/blob/main/data_loading.py) loads the images and then loaded images are split into training & validation sets. The image size is restricted to (256, 256) and all images that do not conform to the  specified size are automatically resized.

### 2. Model
The trained model is made modular by implementing a `FineTuneModel` class.  

Instances of the class can be created by specifying the base model, a flag determining whether the weights of the last few layers should be trained and an offset (i.e.) the number of layers from the end that have to be trained. The available options of the parameters are as follows:

- `model_version`: A string that specifies the base model that should be loaded
    + "ResNet50"


### 3. Train with wandb
```python
sweep_config = {
    "method": "grid",
    "metric": {"name": "val_acc", "goal": "maximize"}, 
    "parameters": {
        "epochs": {"values": [5]},
        "batch_size": {"values": [64, 128]},
        "denselayer_size": {"values": [64, 128]},
        "l_rate": {"values": [0.001, 0.0001]},
        "optimizer": {"values": ["Adam"]},
        "dropout": {"values": [0.2, 0.4]},
        "model_version": {"values": ["resnet50"]},
        "activation": {"values": ["relu", "leakyrelu"]}
    }
}
sweep_id = wandb.sweep(sweep_config, project="Pretrained_Resnet50-Model")# Run the Sweep Agent
wandb.agent(sweep_id, function=train, count=20)
```

### 5. Results
The results of the parameter sweep can be accessed here: [Wandb report Part-B](https://wandb.ai/da24m026-indian-institute-of-technology-madras/Simple_cnn/reports/DA6401-Assignment-2---VmlldzoxMjE3ODE4OQ#question-3)
