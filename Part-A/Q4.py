# Part A -Q4  Running best model obtained from training on test dataset
from utils import *
from simpleCNN import SimpleCNN

################################################################
# Preparing Test Dataset
################################################################

data_module = INaturalistDataModule(
    data_dir='/kaggle/input/inaturalist-dataset/inaturalist_12K',
    batch_size=64,
    use_augmentation=False
)
data_module.setup()

test_loader = data_module.test_dataloader()

###################################################
# Optimal hyperparameters can be specified here
###################################################

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

# load WandB API Key from Kaggle Secrets
user_secrets = UserSecretsClient()
wandb_key = user_secrets.get_secret("wandb")
wandb.login(key=wandb_key)

#  Initialize WandB Project
wandb.init(project="Simple_cnn", name="best_model_eval")

# Load WandB Logger
wandb_logger = WandbLogger(project="Simple_cnn", log_model="all")


trainer = pl.Trainer(
    max_epochs=10,
    logger=wandb_logger,
)

trainer.fit(best_model, data_module)

test_results = trainer.test(best_model, test_loader)
test_accuracy = test_results[0]['test_acc']


wandb.log({"Test Accuracy": test_accuracy})

## Plotting 10x3 predicted images and logging plot to wandb
def log_test_predictions(model, dataloader, num_classes=10, num_per_class=3):
    model.eval()
    class_images = {i: [] for i in range(num_classes)}  # Store images per class

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # Collect 3 images per class
            for img, label, pred in zip(images, labels, preds):
                if len(class_images[label.item()]) < num_per_class:
                    class_images[label.item()].append((img, pred.item()))

            # Stop if we have enough samples
            if all(len(class_images[i]) == num_per_class for i in range(num_classes)):
                break

    # Create a grid
    fig, axes = plt.subplots(num_classes, num_per_class, figsize=(num_per_class * 3, num_classes * 3))
    
    for class_idx, ax_row in enumerate(axes):
        for img_idx, ax in enumerate(ax_row):
            if class_idx in class_images and len(class_images[class_idx]) > img_idx:
                img, pred_label = class_images[class_idx][img_idx]
                img = img.permute(1, 2, 0).cpu().numpy()

                ax.imshow(img)
                ax.set_title(f"Pred: {pred_label}", fontsize=10)
                ax.axis("off")

    plt.tight_layout()

    # Log Grid to WandB
    wandb.log({"Test Predictions": wandb.Image(fig, caption="Per-Class Predictions (10x3)")})
    plt.close(fig)


# Call the function after testing
test_dataloader = data_module.test_dataloader()
log_test_predictions(best_model, test_dataloader)

wandb.finish()


