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
            image = np.float32(image) / 255.0

            gender = row.gender
            class_id = 0 if gender == 'man' else 1
            # class_id = np.array([class_id], dtype=np.float32)

            self.data.append({"image": image, "class_id": class_id})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transforms:
            sample["image"] = self.transforms(image=sample["image"])["image"]
        else:
            sample["image"] = np.transpose(sample["image"], (2, 0, 1))

        return sample["image"], sample["class_id"]