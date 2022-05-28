import torch
import torchvision.transforms
from torch.utils import data
import os
from tqdm import tqdm

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from matplotlib import pyplot as plt

import torchvision.models as models


import moco.builder

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


class ImageDataset(data.Dataset):
    def __init__(self, data_folder, transform, img_suffix='.jpg'):
        self.data_folder = data_folder
        self.img_names = [img_name for img_name in os.listdir(self.data_folder) if img_name.endswith(img_suffix)]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_folder, self.img_names[index])
        image = Image.open(image_path).convert('RGB')
        img = self.transform(image)
        return img, image_path


model = moco.builder.MoCo(
        models.__dict__['resnet50'],
        128, 65536, 0.999, 0.07)

model.load_param('checkpoint_0030.pth.tar')

model.cuda()
model.eval()

transforms = torchvision.transforms.Compose(

[
            torchvision.transforms.Resize(256),

            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
        ]

)

image_dataset = ImageDataset('/mnt/sda1/dataset/FashionAI/imgs', transforms)

image_loader = data.DataLoader(
    image_dataset, batch_size=256, shuffle=False,
    num_workers=4, pin_memory=True, sampler=data.sampler.SequentialSampler(image_dataset)
)

features_list = []
image_path_list = []
for img_batch, image_path in tqdm(image_loader, total=len(image_loader)):
    with torch.no_grad():
        img_batch = img_batch.cuda()
        features = model.encoder_k(img_batch)
        features_list.append(features)
        image_path_list = image_path_list + list(image_path)

features = torch.cat(features_list, dim=0)

features = torch.nn.functional.normalize(features, dim=1)
print(features.shape)
### sim
sim_score = features@features.T
sim_score = sim_score.softmax(-1)

topk = sim_score.topk(10, dim=-1)




