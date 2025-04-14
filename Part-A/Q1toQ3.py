## Part A : Q1 -Q3

from utils import *
from data_loading import INaturalistDataModule
from simpleCNN import SimpleCNN


###############################################
# Listing the hyperparameters in wandb config     
###############################################
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


# Initializing the model architecture and Setting up wandb sweeps


def train():
    # Initialize wandb run
    wandb.init(project="Simple_cnn")  
    config = wandb.config

    # print config to verify keys exist
    print("Loaded Config:", config)

    
    model = SimpleCNN(
         # Default filters
        conv_filters=config.get("conv_filters", [32, 32, 32, 32, 32]), 
        conv_kernel_sizes=config.get("conv_kernel_sizes", [3, 3, 3, 3, 3]),
        conv_activation=config.get("conv_activation", "ReLU"),
        dense_neurons=config.get("dense_neurons", 128),
        dense_activation=config.get("dense_activation", "ReLU"),
        dropout=config.get("dropout", 0.2),
        use_batchnorm=config.get("use_batchnorm", False),
        lr=config.get("lr", 1e-3)
    )

    data_module = INaturalistDataModule(
        data_dir='/kaggle/input/inaturalist-dataset/inaturalist_12K',
        batch_size=config.get("batch_size", 32),
        use_augmentation=config.get("use_augmentation", False) 
    )
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # Log training process with wandb
    wandb_logger = WandbLogger(project="Simple_cnn", log_model="all")

    # Define Trainer pytorch_lighting method which automatically trains model and logs metrics 
    trainer = pl.Trainer(
        max_epochs=10,    
        logger=wandb_logger,
    )

    # Train the model
    trainer.fit(model, train_loader,val_loader)
 
if __name__ == "__main__":
    
    user_secrets = UserSecretsClient() #if using in  kaggle set up the wandb key and use that key to access wandb 
    secret_value_0 = user_secrets.get_secret("wandb")
    wandb.login(key=secret_value_0)
    sweep_id = wandb.sweep(sweep_config,project='Simple_cnn')
    
    wandb.agent(sweep_id, train, count=30)  # Runs 30 experiments




