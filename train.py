import torch
import os
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torch.utils.data import DataLoader
from torch.nn.functional import grid_sample
from tqdm import tqdm

from models.main_model import JED_VCOD_Fauna_Simplified
from datasets.moca_video_dataset import MoCAVideoDataset
from utils.losses import DiceLoss

def main():
    # 1. 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JED_VCOD_Fauna_Simplified().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # RAFT 모델 로드
    raft_weights = Raft_Large_Weights.DEFAULT
    raft_transforms = raft_weights.transforms()
    raft_model = raft_large(weights=raft_weights, progress=False).to(device)
    raft_model.eval()

    bce_loss = torch.nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()
    l1_loss = torch.nn.L1Loss()
    lambda_temporal = 0.1
    num_epochs = 50
    batch_size = 2

    # 2. 데이터 로더 설정
    train_dataset = MoCAVideoDataset(
        synthetic_data_root='data/Night-Camo-Fauna/',
        annotation_file='data/MoCA/Annotations/annotations_modified.csv',
        clip_len=8
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 3. 학습 루프
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for video_clip, ground_truth_masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            video_clip = video_clip.to(device)
            ground_truth_masks = ground_truth_masks.to(device)

            optimizer.zero_grad()

            predicted_masks = model(video_clip)

            loss_seg = bce_loss(predicted_masks, ground_truth_masks) + dice_loss(predicted_masks, ground_truth_masks)

            loss_temporal = 0
            b, t, c, h, w = video_clip.shape

            if t > 1:
                img1_batch = video_clip[:, :-1].reshape(-1, c, h, w)
                img2_batch = video_clip[:, 1:].reshape(-1, c, h, w)

                img1_transformed, img2_transformed = raft_transforms(img1_batch, img2_batch)

                with torch.no_grad():
                    flows = raft_model(img1_transformed, img2_transformed)[-1]

                flows_unbatched = flows.view(b, t - 1, 2, h, w)

                for i in range(t - 1):
                    flow_i = flows_unbatched[:, i]

                    grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
                    grid = torch.stack((grid_x, grid_y), 2).float()
                    grid.requires_grad = False

                    # RAFT flow는 (x, y) 순서의 변위(displacement)
                    displacement = flow_i.permute(0, 2, 3, 1)
                    warped_grid = grid + displacement

                    # grid_sample을 위해 [-1, 1] 범위로 정규화
                    warped_grid[..., 0] = 2.0 * warped_grid[..., 0] / (w - 1) - 1.0
                    warped_grid[..., 1] = 2.0 * warped_grid[..., 1] / (h - 1) - 1.0

                    mask_t_warped = grid_sample(predicted_masks[:, i], warped_grid, mode='bilinear', padding_mode='border', align_corners=False)
                    loss_temporal += l1_loss(mask_t_warped, predicted_masks[:, i+1])

            total_loss = loss_seg + lambda_temporal * (loss_temporal / (t - 1) if t > 1 else 0)
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_loader)}")

        if (epoch + 1) % 5 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/jed_vcod_epoch_{epoch+1}.pth")

if __name__ == '__main__':
    main()