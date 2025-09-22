import torch
import torch.nn as nn
import os
import logging
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torch.utils.data import DataLoader, random_split
from torch.nn.functional import grid_sample
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from config import cfg
from models.main_model import JED_VCOD_Fauna_Simplified
from datasets.moca_video_dataset import MoCAVideoDataset
from utils.losses import DiceLoss, FocalLoss # FocalLoss import
from utils.logger import setup_logger

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def train_one_epoch(model, raft_model, raft_transforms, train_loader, optimizer, device, focal_loss, dice_loss, l1_loss, writer, epoch, train_cfg):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
    for i, (video_clip, ground_truth_masks) in enumerate(progress_bar):
        video_clip = video_clip.to(device)
        ground_truth_masks = ground_truth_masks.to(device)

        optimizer.zero_grad()
        predicted_masks = model(video_clip)
        
        # BCE Loss를 Focal Loss로 교체
        loss_focal = focal_loss(predicted_masks, ground_truth_masks)
        loss_dice = dice_loss(predicted_masks, ground_truth_masks)
        loss_seg = loss_focal + train_cfg['dice_weight'] * loss_dice

        loss_temporal = 0
        b, t, c, h, w = video_clip.shape
        if t > 1 and train_cfg['lambda_temporal'] > 0:
            img1_batch = video_clip[:, :-1].reshape(-1, c, h, w)
            img2_batch = video_clip[:, 1:].reshape(-1, c, h, w)
            img1_transformed, img2_transformed = raft_transforms(img1_batch, img2_batch)
            with torch.no_grad():
                flows = raft_model(img1_transformed, img2_transformed)[-1]
            flows_unbatched = flows.view(b, t - 1, 2, h, w)

            for frame_idx in range(t - 1):
                flow_i = flows_unbatched[:, frame_idx]
                grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
                grid = torch.stack((grid_x, grid_y), 2).float()
                displacement = flow_i.permute(0, 2, 3, 1)
                warped_grid = grid + displacement
                warped_grid[..., 0] = 2.0 * warped_grid[..., 0] / (w - 1) - 1.0
                warped_grid[..., 1] = 2.0 * warped_grid[..., 1] / (h - 1) - 1.0
                mask_t_warped = grid_sample(predicted_masks[:, frame_idx], warped_grid, mode='bilinear', padding_mode='border', align_corners=False)
                loss_temporal += l1_loss(mask_t_warped, predicted_masks[:, frame_idx+1])
        
        loss_temporal_avg = loss_temporal / (t - 1) if t > 1 and t > 1 else torch.tensor(0.0).to(device)
        total_loss = loss_seg + train_cfg['lambda_temporal'] * loss_temporal_avg
        total_loss.backward()
        optimizer.step()
        
        current_loss = total_loss.item()
        epoch_loss += current_loss
        
        global_step = epoch * len(train_loader) + i
        writer.add_scalar('Loss/total_step', current_loss, global_step)
        writer.add_scalar('Loss/segmentation_step', loss_seg.item(), global_step)
        if t > 1:
            writer.add_scalar('Loss/temporal_step', loss_temporal_avg.item(), global_step)
            
    return epoch_loss / len(train_loader)

def validate(model, val_loader, device, focal_loss, dice_loss, train_cfg):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for video_clip, ground_truth_masks in tqdm(val_loader, desc="Validating"):
            video_clip = video_clip.to(device)
            ground_truth_masks = ground_truth_masks.to(device)
            predicted_masks = model(video_clip)
            
            # BCE Loss를 Focal Loss로 교체
            loss_focal = focal_loss(predicted_masks, ground_truth_masks)
            loss_dice = dice_loss(predicted_masks, ground_truth_masks)
            loss = loss_focal + train_cfg['dice_weight'] * loss_dice
            val_loss += loss.item()
    return val_loss / len(val_loader)

def main():
    common_cfg = cfg.common
    train_cfg = cfg.train
    
    logger = setup_logger(train_cfg['log_dir'], train_cfg['experiment_name'])
    writer = SummaryWriter(os.path.join(train_cfg['log_dir'], 'tensorboard', train_cfg['experiment_name']))

    if common_cfg['gpu_ids']:
        os.environ["CUDA_VISIBLE_DEVICES"] = common_cfg['gpu_ids']
        gpu_ids = list(range(len(common_cfg['gpu_ids'].split(','))))
        device = torch.device(f"cuda:{gpu_ids[0]}")
    else:
        gpu_ids = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = JED_VCOD_Fauna_Simplified()
    model.apply(weights_init)
    logger.info("Model weights initialized with Kaiming Normal.")

    if gpu_ids and len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=train_cfg['scheduler_factor'], patience=train_cfg['scheduler_patience'])

    raft_weights = Raft_Large_Weights.DEFAULT
    raft_transforms = raft_weights.transforms()
    raft_model = raft_large(weights=raft_weights, progress=False).to(device)
    raft_model.eval()

    # BCE Loss를 Focal Loss로 교체
    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    l1_loss = nn.L1Loss()

    full_dataset = MoCAVideoDataset(
        synthetic_data_root=train_cfg['data_root'],
        annotation_file=train_cfg['annotation_file'],
        clip_len=common_cfg['clip_len']
    )
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=common_cfg['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=common_cfg['num_workers'])
    logger.info(f"Dataset loaded: {len(train_dataset)} for training, {len(val_dataset)} for validation.")

    best_val_loss = np.inf
    early_stop_counter = 0

    logger.info("--- Starting Training ---")
    for epoch in range(train_cfg['epochs']):
        train_loss = train_one_epoch(model, raft_model, raft_transforms, train_loader, optimizer, device, focal_loss, dice_loss, l1_loss, writer, epoch, train_cfg)
        val_loss = validate(model, val_loader, device, focal_loss, dice_loss, train_cfg)
        
        logger.info(f"Epoch {epoch+1}/{train_cfg['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        writer.add_scalar('Loss/val_epoch', val_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            os.makedirs(train_cfg['checkpoint_dir'], exist_ok=True)
            save_path = os.path.join(train_cfg['checkpoint_dir'], train_cfg['checkpoint_name'])
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, save_path)
            logger.info(f"Validation loss improved. Saved best model to {save_path}")
        else:
            early_stop_counter += 1
            logger.info(f"Validation loss did not improve. Counter: {early_stop_counter}/{train_cfg['patience']}")

        if early_stop_counter >= train_cfg['patience']:
            logger.info("Early stopping triggered.")
            break

    writer.close()
    logger.info("--- Training Finished ---")

if __name__ == '__main__':
    main()