from binaryclassificationnet import BinaryClassificationNet
from binaryinception import BinaryInception
from binaryresnet import BinaryResnet
from dataset import GenderRecognitionDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import mlflow.pytorch

# Set hyperparameters
BATCH_SIZE = 16
NUM_WORKERS = 8
NUM_EPOCHS = 100
CSV_DIR = 'faces.csv'
MIN_DELTA = 1e-4
PATIENCE = 5
LEARNING_RATE = 5e-4
LOSS = nn.BCEWithLogitsLoss()

# Set random seed
seed = 42

# Set transforms
transforms = A.Compose([
    A.Rotate(limit=90, p=0.8),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomBrightnessContrast(),
    ToTensorV2(),
])

# Load data
df = pd.read_csv(CSV_DIR)
train_df, val_df = train_test_split(df, test_size=0.3, random_state=seed)
val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=seed)

train_dataset = GenderRecognitionDataset(train_df, transforms=transforms)
val_dataset = GenderRecognitionDataset(val_df)
test_dataset = GenderRecognitionDataset(test_df)

# Get dataloaders
train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=NUM_WORKERS,
                          drop_last=True)
val_loader = DataLoader(val_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        num_workers=NUM_WORKERS,
                        drop_last=True)

# Enable autologging
mlflow.pytorch.autolog()
mlflow.log_params({'batch_size': BATCH_SIZE, 'loss': LOSS})

# Create model
# model = BinaryClassificationNet(loss=LOSS, lr=LEARNING_RATE)
model = BinaryResnet(loss=LOSS, lr=LEARNING_RATE)

# Log model name as a tag
mlflow.set_tag('model_name', model.__class__.__name__)

# Create callbacks
early_stopping = EarlyStopping('val_loss',
                               min_delta=MIN_DELTA,
                               patience=PATIENCE,
                               verbose=True)

checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', verbose=True)

# Train model
trainer = pl.Trainer(max_epochs=NUM_EPOCHS, callbacks=[early_stopping, checkpoint_callback])
trainer.fit(model, train_loader, val_loader)

# TODO: Add test set evaluation and log test results