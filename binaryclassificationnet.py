import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

"""
This is a generic binary classification model that can be used for any binary classification task.
It takes in an image of size 3x299x299 and outputs a binary classification of 0 or 1.
"""

class BinaryClassificationNet(pl.LightningModule):
    def __init__(self, loss=nn.BCEWithLogitsLoss(), lr=1e-3):
        super(BinaryClassificationNet, self).__init__()
        
        # Loss and LR
        self.loss = loss
        self.lr = lr

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 37 * 37, 512)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Add dropout for regularization
        
        self.fc2 = nn.Linear(512, 1)
        # Define loss function and metrics
        self.criterion = self.loss
        self.accuracy = torchmetrics.Accuracy(task='binary')

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        labels = torch.Tensor(labels).unsqueeze(1)
        labels = labels.float()
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        self.log('train_acc', self.accuracy(outputs, labels), prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        labels = torch.Tensor(labels).unsqueeze(1)
        labels = labels.float()
        outputs = self(inputs)
        # preds = (outputs > 0.5).float()
        loss = self.criterion(outputs, labels)

        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', self.accuracy(outputs, labels), on_epoch=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
