import os
import shutil
from tqdm import tqdm

save_path = r'/mnt/sda1/dataset/FashionAI/imgs'

img_root_path_list = ['/mnt/sda1/dataset/FashionAI/fashionAI_attributes_train1/Images', '/mnt/sda1/dataset/FashionAI/fashionAI_attributes_train2/Images']
for img_root_path in img_root_path_list:
    for folder in os.listdir(img_root_path):
        for img_name in tqdm(os.listdir(os.path.join(img_root_path, folder))):
            shutil.copy(os.path.join(img_root_path, folder, img_name), os.path.join(save_path, img_name))
