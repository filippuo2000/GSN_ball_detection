from torchvision.io import read_image
from torch.utils.data import Dataset
import torch.nn.functional as F

class TrackNetDataset(Dataset):
    def __init__(self, labels, img_dirs, size, transform=None, target_transform=None):
        self.img_labels = labels
        self.img_dirs = img_dirs
        self.size = size
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, idx):
        img_path = self.img_dirs[idx]
        image = read_image(img_path)
        image = image.float()
        image = F.interpolate(image.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False).squeeze(0)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
