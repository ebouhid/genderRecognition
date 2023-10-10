from binaryclassificationnet import BinaryClassificationNet
from dataset import GenderRecognitionDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

BATCH_SIZE = 64
NUM_WORKERS = 8
NUM_EPOCHS = 100
CSV_DIR = 'faces.csv'
MIN_DELTA = 1e-2
PATIENCE = 5

# Set random seed
seed = 42

# Load data
df = pd.read_csv(CSV_DIR)
train_df, val_df = train_test_split(df, test_size=0.3, random_state=seed)
val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=seed)

train_dataset = GenderRecognitionDataset(train_df)
val_dataset = GenderRecognitionDataset(val_df)
test_dataset = GenderRecognitionDataset(test_df)

# Get dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Create model
model = BinaryClassificationNet()

# Create early stopping callback
early_stopping = EarlyStopping('val_accuracy', min_delta=MIN_DELTA,patience=PATIENCE)

# Train model
trainer = pl.Trainer(max_epochs=NUM_EPOCHS)
trainer.fit(model, train_loader, val_loader)