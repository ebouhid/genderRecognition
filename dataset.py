import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class GenderRecognitionDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms

        # Load all images into memory
        self.data = []

        for idx, row in self.df.iterrows():
            image = Image.open(row.filepath)
            image = image.resize((299, 299))
            image = np.array(image)
            image = image.transpose(2, 0, 1)
            image = np.float32(image) / 255.0
            
            gender = row.gender
            class_id = 0 if gender == 'man' else 1

            self.data.append({"image": image, "class_id": class_id})
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transforms:
            raise NotImplementedError
        
        return sample["image"], sample["class_id"]