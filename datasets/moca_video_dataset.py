import os
import cv2
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class MoCAVideoDataset(Dataset):
    def __init__(self, synthetic_data_root, annotation_file, clip_len=8, resolution=(224, 224)):
        self.synthetic_data_root = synthetic_data_root
        self.clip_len = clip_len
        self.resolution = resolution
        self.transform = transforms.Compose([
            transforms.Resize(self.resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(self.resolution),
            transforms.ToTensor(),
        ])
        
        # ▼▼▼ 수정된 부분 ▼▼▼
        # 아무 옵션 없이 깔끔한 CSV 파일을 바로 읽어옵니다.
        df = pd.read_csv(annotation_file)
        
        # 'filename' 대신 'file_list' 열을 사용해서 'video_name'을 만듭니다.
        df['video_name'] = df['file_list'].apply(lambda x: os.path.dirname(x))
        self.annotations_df = df

        self.clips = []
        video_groups = self.annotations_df.groupby('video_name')
        for _, group in video_groups:
            # ▼▼▼ 수정된 부분 ▼▼▼
            # 정렬 기준 열을 'file_list'로 변경합니다.
            sorted_group = group.sort_values(by='file_list')
            if len(sorted_group) >= clip_len:
                for i in range(0, len(sorted_group) - clip_len + 1, clip_len):
                    clip_info = sorted_group.iloc[i : i + clip_len]
                    self.clips.append(clip_info)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_info = self.clips[idx]

        image_clip = []
        mask_clip = []

        for _, row in clip_info.iterrows():
            # ▼▼▼ 수정된 부분 ▼▼▼
            # 파일명을 가져올 때 'file_list' 열을 사용합니다.
            base_filename = os.path.splitext(os.path.basename(row['file_list']))[0]
            # synthetic_data_root를 사용하지 않고 file_list의 절대 경로를 그대로 사용합니다.
            img_path = row['file_list']

            try:
                # 경로에 .jpg가 포함되어 있을 수 있으니 .png로 강제 변경하지 않습니다.
                image = Image.open(img_path).convert("RGB")
                width, height = image.size
            except FileNotFoundError:
                print(f"Warning: File not found {img_path}. Returning empty tensor.")
                return torch.zeros(self.clip_len, 3, *self.resolution), torch.zeros(self.clip_len, 1, *self.resolution)

            mask = np.zeros((height, width), dtype=np.uint8)
            try:
                # spatial_coordinates 열을 사용하도록 변경합니다.
                if pd.notna(row['spatial_coordinates']) and row['spatial_coordinates'] != '[]':
                    # 문자열 '[]'를 제거하고 숫자로 변환합니다.
                    coords_str = row['spatial_coordinates'].strip('[]')
                    coords = list(map(int, coords_str.split(',')))
                    
                    # shape_id, x, y, width, height 정보를 추출합니다.
                    shape_id = coords[0]
                    if shape_id == 2: # Rectangle
                        x, y, w, h = coords[1], coords[2], coords[3], coords[4]
                        # 다각형 형식으로 변환하여 마스크를 채웁니다.
                        polygon = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
                        cv2.fillPoly(mask, [polygon], 1)

            except (ValueError, KeyError, IndexError):
                # JSON 파싱 대신 다른 예외를 처리합니다.
                pass

            image_clip.append(self.transform(image))
            mask_clip.append(self.mask_transform(Image.fromarray(mask)))
        
        if not image_clip:
             return torch.zeros(self.clip_len, 3, *self.resolution), torch.zeros(self.clip_len, 1, *self.resolution)

        # (T, C, H, W) 형태로 반환되도록 stack의 dim을 0으로 설정
        return torch.stack(image_clip, dim=0), torch.stack(mask_clip, dim=0)