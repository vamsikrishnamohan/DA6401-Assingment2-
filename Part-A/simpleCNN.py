#####################################
# Initializing the model architecture 
# Simple CNN Model with flexibile  conv blocks, each: Conv -> (optional BatchNorm) -> Activation -> MaxPool
#####################################
from utils import * 

class SimpleCNN(pl.LightningModule):
    def __init__(self, conv_filters=[32, 32, 32, 32, 32],
                 conv_kernel_sizes=[3, 3, 3, 3, 3],
                 conv_activation='ReLU',
                 dense_neurons=128,
                 dense_activation='ReLU',
                 dropout=0.0,
                 use_batchnorm=False,
                 lr=1e-3):
        super(SimpleCNN, self).__init__()
        self.save_hyperparameters()
        # Dictionary mapping activation names to modules
        activations = {
            'ReLU': nn.ReLU,
            'GELU': nn.GELU,
            'SiLU': nn.SiLU
            
        }
        conv_act = activations.get(conv_activation, nn.ReLU)
        dense_act = activations.get(dense_activation, nn.ReLU)
        
        self.conv_layers = nn.ModuleList()
        in_channels = 3  # iNaturalist images have 3 channels (RGB)
        for i in range(5):
            out_channels = conv_filters[i]
            kernel_size = conv_kernel_sizes[i]
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
            block = [conv]
            if use_batchnorm:
                block.append(nn.BatchNorm2d(out_channels))
            block.append(conv_act())
            block.append(nn.MaxPool2d(2))
            self.conv_layers.append(nn.Sequential(*block))
            in_channels = out_channels
        
        # Compute the flattened feature size after conv layers
        # Assume a default input image size of 224x224
        dummy_input = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            features = dummy_input
            for layer in self.conv_layers:
                features = layer(features)
            self.flattened_size = features.view(1, -1).shape[1]
        
        # Dense layer followed by the output layer (10 neurons for 10 classes)
        self.dense = nn.Linear(self.flattened_size, dense_neurons)
        self.dense_activation = dense_act()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.out = nn.Linear(dense_neurons, 10)
        self.lr = lr

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = torch.flatten(x, 1)
        x = self.dense(x)
        x = self.dense_activation(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.out(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('test_loss', loss)
        self.log('test_acc', acc)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer