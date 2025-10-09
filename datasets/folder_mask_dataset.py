import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random

class FolderImageMaskDataset(Dataset):
    def __init__(self, root_dir, image_folder_name, mask_folder_name, clip_len=8, resolution=(224, 224), is_train=True, use_augmentation=True):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.resolution = resolution
        self.is_train = is_train
        self.use_augmentation = use_augmentation
        
        self.image_paths = []
        self.mask_paths = []

        print(f"Scanning dataset in '{root_dir}'...")
        for dirpath, dirnames, filenames in os.walk(root_dir):
            if os.path.basename(dirpath) == image_folder_name:
                for filename in sorted(filenames):
                    if filename.endswith(('.jpg', '.png', '.jpeg')):
                        image_path = os.path.join(dirpath, filename)
                        mask_path = image_path.replace(f'/{image_folder_name}/', f'/{mask_folder_name}/').replace(f'\\{image_folder_name}\\', f'\\{mask_folder_name}\\')
                        mask_path = os.path.splitext(mask_path)[0] + '.png'
                        if os.path.exists(mask_path):
                            self.image_paths.append(image_path)
                            self.mask_paths.append(mask_path)
        
        print(f"Found {len(self.image_paths)} image-mask pairs.")
        
        self.image_transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        try:
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
        except FileNotFoundError:
            return None

        # is_train 모드이고, use_augmentation이 True일 때만 데이터 증강을 적용합니다.
        if self.is_train and self.use_augmentation:
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            
            image = self.color_jitter(image)

        image_tensor = self.image_transform(image)
        mask_tensor = self.mask_transform(mask)
        
        image_clip = torch.stack([image_tensor] * self.clip_len, dim=0)
        mask_clip = torch.stack([mask_tensor] * self.clip_len, dim=0)

        return image_clip, mask_clip