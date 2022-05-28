from torch.utils import data
import os

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class MoCoDataset(data.Dataset):
    def __init__(self, data_folder, transform, img_suffix='.jpg'):
        self.data_folder = data_folder
        self.img_names = [img_name for img_name in os.listdir(self.data_folder) if img_name.endswith(img_suffix)]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_folder, self.img_names[index])
        image = Image.open(image_path).convert('RGB')
        img_q = self.transform(image)
        img_k = self.transform(image)
        return img_q, img_k
