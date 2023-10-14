import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl

class BinaryResnet(pl.LightningModule):
    def __init__(self, loss=nn.BCEWithLogitsLoss(),lr=1e-3):
        super(BinaryResnet, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)

        # Remove the original final classification layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  
        
        self.resnet.fc = nn.Linear(num_features, 1)
        
        # Define loss function and metrics
        self.criterion = loss
        self.accuracy = torchmetrics.Accuracy(task='binary')

        # Define learning rate
        self.lr = lr

    def forward(self, x):
        outputs  = self.resnet(x)
        return outputs

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        labels = torch.Tensor(labels).unsqueeze(1)
        labels = labels.float()
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.accuracy(outputs, labels), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        labels = torch.Tensor(labels).unsqueeze(1)
        labels = labels.float()
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        self.log('val_loss', loss)
        self.log('val_acc', self.accuracy(outputs, labels))
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
