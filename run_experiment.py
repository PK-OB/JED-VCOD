import torch
import torch.nn as nn
import os
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torch.utils.data import DataLoader, random_split, Subset
from torch.nn.functional import grid_sample
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

from config import cfg
from models.main_model import JED_VCOD_Fauna_Simplified
from datasets.moca_video_dataset import MoCAVideoDataset
from datasets.folder_mask_dataset import FolderImageMaskDataset
from utils.losses import DiceLoss, FocalLoss
from utils.logger import setup_logger

def unnormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def verify_and_save_samples(dataset, common_cfg, train_cfg):
    logging.info("--- Verifying dataset samples ---")
    if len(dataset) < 5:
        logging.warning("Dataset has fewer than 5 samples, skipping verification.")
        return

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Dataset Verification: 5 Random Samples', fontsize=16)
    random_indices = random.sample(range(len(dataset)), 5)

    for i, idx in enumerate(random_indices):
        data_sample = dataset[idx]
        if data_sample is None: continue
        
        video_clip, mask_clip = data_sample
        frame_index_to_show = common_cfg['clip_len'] // 2
        
        image_tensor = video_clip[frame_index_to_show]
        image_to_show = unnormalize(image_tensor).numpy().transpose(1, 2, 0)
        axes[0, i].imshow(np.clip(image_to_show, 0, 1))
        axes[0, i].set_title(f"Sample index {idx} - Image")
        axes[0, i].axis('off')

        mask_tensor = mask_clip[frame_index_to_show]
        mask_to_show = mask_tensor.squeeze().numpy()
        axes[1, i].imshow(mask_to_show, cmap='gray')
        axes[1, i].set_title(f"Sample index {idx} - Mask")
        axes[1, i].axis('off')

    debug_dir = os.path.join(train_cfg['log_dir'], 'debug_images')
    os.makedirs(debug_dir, exist_ok=True)
    save_path = os.path.join(debug_dir, f"{train_cfg['experiment_name']}_dataset_verification.png")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close(fig)
    logging.info(f"Verification image saved to: {save_path}")

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
    for i, batch in enumerate(progress_bar):
        if batch is None: continue
        video_clip, ground_truth_masks = batch
        
        video_clip = video_clip.to(device)
        ground_truth_masks = ground_truth_masks.to(device)

        optimizer.zero_grad()
        predicted_masks = model(video_clip)
        
        loss_focal = focal_loss(predicted_masks, ground_truth_masks)
        loss_dice = dice_loss(predicted_masks, ground_truth_masks)
        loss_seg = loss_focal + train_cfg['dice_weight'] * loss_dice

        loss_temporal = 0
        b, t, c, h, w = video_clip.shape
        if t > 1 and train_cfg['lambda_temporal'] > 0:
            img1_batch = video_clip[:, :-1].reshape(-1, c, h, w)
            img2_batch = video_clip[:, 1:].reshape(-1, c, h, w)
            if not torch.equal(img1_batch, img2_batch):
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
        
        loss_temporal_avg = loss_temporal / (t - 1) if t > 1 else torch.tensor(0.0).to(device)
        total_loss = loss_seg + train_cfg['lambda_temporal'] * loss_temporal_avg
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logging.error("!!! Loss is NaN or Inf. Stopping training.")
            return -1

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        current_loss = total_loss.item()
        epoch_loss += current_loss
        
        global_step = epoch * len(train_loader) + i
        writer.add_scalar('Loss/total_step', current_loss, global_step)

    if (epoch + 1) % train_cfg['debug_image_interval'] == 0 and 'predicted_masks' in locals() and len(predicted_masks) > 0:
        pred_to_save = torch.sigmoid(predicted_masks[0, 0]).cpu()
        gt_to_save = ground_truth_masks[0, 0].cpu()
        
        debug_dir = os.path.join(train_cfg['log_dir'], 'debug_images')
        os.makedirs(debug_dir, exist_ok=True)
        save_image(pred_to_save, os.path.join(debug_dir, f'epoch_{epoch+1}_prediction.png'))
        save_image(gt_to_save, os.path.join(debug_dir, f'epoch_{epoch+1}_ground_truth.png'))
        
        writer.add_image('Images/Prediction', pred_to_save, epoch)
        writer.add_image('Images/GroundTruth', gt_to_save, epoch)

    return epoch_loss / len(train_loader) if len(train_loader) > 0 else 0

def validate(model, val_loader, device, focal_loss, dice_loss, train_cfg):
    model.eval()
    val_loss = 0
    num_batches = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            if batch is None: continue
            video_clip, ground_truth_masks = batch
            video_clip = video_clip.to(device)
            ground_truth_masks = ground_truth_masks.to(device)
            predicted_masks = model(video_clip)
            
            loss_focal = focal_loss(predicted_masks, ground_truth_masks)
            loss_dice = dice_loss(predicted_masks, ground_truth_masks)
            loss = loss_focal + train_cfg['dice_weight'] * loss_dice
            val_loss += loss.item()
            num_batches += 1
            
    return val_loss / num_batches if num_batches > 0 else 0

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg['lr'], weight_decay=train_cfg['weight_decay'])
    logger.info(f"Optimizer: AdamW with lr={train_cfg['lr']}, weight_decay={train_cfg['weight_decay']}")
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=train_cfg['scheduler_factor'], patience=train_cfg['scheduler_patience'])

    raft_weights = Raft_Large_Weights.DEFAULT
    raft_transforms = raft_weights.transforms()
    raft_model = raft_large(weights=raft_weights, progress=False).to(device)
    raft_model.eval()

    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    l1_loss = nn.L1Loss()
    
    if train_cfg['dataset_type'] == 'csv':
        full_dataset = MoCAVideoDataset(
            synthetic_data_root=train_cfg['csv_data_root'],
            annotation_file=train_cfg['annotation_file'],
            clip_len=common_cfg['clip_len']
        )
        logger.info(f"Loading CSV-based dataset from: {train_cfg['annotation_file']}")
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    elif train_cfg['dataset_type'] == 'folder':
        train_dataset_full = FolderImageMaskDataset(
            root_dir=train_cfg['folder_data_root'],
            image_folder_name=train_cfg['image_folder_name'],
            mask_folder_name=train_cfg['mask_folder_name'],
            clip_len=common_cfg['clip_len'],
            is_train=True,
            use_augmentation=train_cfg['use_augmentation']
        )
        val_dataset_full = FolderImageMaskDataset(
            root_dir=train_cfg['folder_data_root'],
            image_folder_name=train_cfg['image_folder_name'],
            mask_folder_name=train_cfg['mask_folder_name'],
            clip_len=common_cfg['clip_len'],
            is_train=False,
            use_augmentation=False
        )
        logger.info(f"Loading folder-based dataset from: {train_cfg['folder_data_root']}")
        logger.info(f"Data Augmentation: {train_cfg['use_augmentation']}")

        total_size = len(train_dataset_full)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        indices = list(range(total_size))
        random.shuffle(indices)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_dataset = Subset(train_dataset_full, train_indices)
        val_dataset = Subset(val_dataset_full, val_indices)
    else:
        raise ValueError(f"Invalid dataset_type in config: {train_cfg['dataset_type']}")

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if not batch: return None
        return torch.utils.data.dataloader.default_collate(batch)

    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=common_cfg['num_workers'], collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=common_cfg['num_workers'], collate_fn=collate_fn)
    logger.info(f"Dataset split: {len(train_dataset)} for training, {len(val_dataset)} for validation.")
    
    verify_and_save_samples(train_dataset, common_cfg, train_cfg)

    best_val_loss = np.inf
    early_stop_counter = 0

    logger.info("--- Starting Training ---")
    for epoch in range(train_cfg['epochs']):
        train_loss = train_one_epoch(model, raft_model, raft_transforms, train_loader, optimizer, device, focal_loss, dice_loss, l1_loss, writer, epoch, train_cfg)
        
        if train_loss == -1: break
            
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