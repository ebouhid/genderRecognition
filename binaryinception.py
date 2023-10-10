import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl

class BinaryInceptionModule(pl.LightningModule):
    def __init__(self, pretrained_inception, learning_rate=1e-3):
        super(BinaryInceptionModule, self).__init__()
        inception = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)

        # Remove the original final classification layer
        inception.fc = nn.Identity()  

        # Add a new binary classification layer
        num_features = inception.AuxLogits.fc.in_features
        binary_classifier = nn.Linear(num_features, 2)  # Two output units for binary classification
        inception.fc = binary_classifier
        
        # Load the pre-trained Inception model and modify it
        self.inception = pretrained_inception
        num_features = self.inception.fc.in_features
        self.inception.fc = nn.Linear(num_features, 2)  # Binary classification layer
        
        # Define loss function and metrics
        self.criterion = nn.BCEWithLogitsLoss()
        self.accuracy = torchmetrics.Accuracy(task='binary')

        # Define learning rate
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.inception(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = (outputs > 0).float()  # Convert logits to binary predictions
        self.log('val_loss', loss)
        self.log('val_acc', self.accuracy(preds, labels))
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
