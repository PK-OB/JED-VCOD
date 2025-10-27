# pk-ob/jed-vcod/JED-VCOD-cc543b29cefb3a45b940bfd01f42c33af7a6bb25/run_experiment.py

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
import torch.nn.functional as F # <-- ▼▼▼ 이 줄이 추가되었습니다! ▼▼▼
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

from config import cfg
from models.main_model import JED_VCOD_Fauna_Simplified
from datasets.moca_video_dataset import MoCAVideoDataset
from datasets.folder_mask_dataset import FolderImageMaskDataset
from utils.losses import DiceLoss, FocalLoss
from utils.logger import setup_logger
from utils.cutmix import cutmix_data # 수정된 cutmix 임포트

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
        
    # ▼▼▼ 수정된 부분: 3x5 (야간/마스크/주간) 플롯으로 변경 ▼▼▼
    fig, axes = plt.subplots(3, 5, figsize=(20, 12)) 
    fig.suptitle('Dataset Verification: 5 Random Samples (Night / Mask / Day)', fontsize=16)
    random_indices = random.sample(range(len(dataset)), 5)
    
    for i, idx in enumerate(random_indices):
        # Subset에서 실제 데이터셋의 아이템을 가져옵니다.
        if isinstance(dataset, Subset):
            data_sample = dataset.dataset[dataset.indices[idx]]
        else:
            data_sample = dataset[idx]
            
        if data_sample is None: continue
        
        # ▼▼▼ 수정된 부분: 3개 항목 수신 ▼▼▼
        video_clip, mask_clip, original_day_clip = data_sample
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        
        frame_index_to_show = common_cfg['clip_len'] // 2
        
        # 1. 야간 이미지
        image_tensor = video_clip[frame_index_to_show]
        image_to_show = unnormalize(image_tensor).numpy().transpose(1, 2, 0)
        axes[0, i].imshow(np.clip(image_to_show, 0, 1))
        axes[0, i].set_title(f"Sample index {idx} - Night Image")
        axes[0, i].axis('off')
        
        # 2. 마스크
        mask_tensor = mask_clip[frame_index_to_show]
        mask_to_show = mask_tensor.squeeze().numpy()
        axes[1, i].imshow(mask_to_show, cmap='gray')
        axes[1, i].set_title(f"Sample index {idx} - Mask")
        axes[1, i].axis('off')
        
        # ▼▼▼ 수정된 부분: 3. 주간 이미지 ▼▼▼
        day_image_tensor = original_day_clip[frame_index_to_show]
        day_image_to_show = day_image_tensor.numpy().transpose(1, 2, 0) # 0~1 범위
        axes[2, i].imshow(np.clip(day_image_to_show, 0, 1))
        axes[2, i].set_title(f"Sample index {idx} - Day Image (GT)")
        axes[2, i].axis('off')
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    debug_dir = os.path.join(train_cfg['log_dir'], 'debug_images')
    os.makedirs(debug_dir, exist_ok=True)
    save_path = os.path.join(debug_dir, f"{train_cfg['experiment_name']}_dataset_verification.png")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close(fig)
    logging.info(f"Verification image saved to: {save_path}")

def train_one_epoch(model, raft_model, raft_transforms, train_loader, optimizer, device, focal_loss, dice_loss, l1_loss, writer, epoch, train_cfg):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
    for i, batch in enumerate(progress_bar):
        if batch is None: continue
        
        # ▼▼▼ 수정된 부분: 3개 항목 수신 ▼▼▼
        video_clip, ground_truth_masks, original_day_images = batch
        
        video_clip = video_clip.to(device)
        ground_truth_masks = ground_truth_masks.to(device)
        original_day_images = original_day_images.to(device) # <-- 추가
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        
        b, t, c, h, w = video_clip.shape
        images_flat = video_clip.view(b*t, c, h, w)
        masks_flat = ground_truth_masks.view(b*t, 1, h, w)
        
        # ▼▼▼ 수정된 부분: 주간 이미지도 flatten ▼▼▼
        original_images_flat = original_day_images.view(b*t, c, h, w)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        r = np.random.rand(1)
        if train_cfg['use_cutmix'] and train_cfg['cutmix_beta'] > 0 and r < train_cfg['cutmix_prob']:
            # ▼▼▼ 수정된 부분: 3개 텐서로 cutmix 수행 ▼▼▼
            images_flat, masks_flat, original_images_flat = cutmix_data(
                images_flat, 
                masks_flat, 
                original_images_flat, 
                train_cfg['cutmix_beta'], 
                use_cuda=True
            )
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        
        video_clip = images_flat.view(b, t, c, h, w)
        ground_truth_masks = masks_flat.view(b, t, 1, h, w)
        # ▼▼▼ 수정된 부분: 주간 이미지도 view 복원 ▼▼▼
        original_day_images = original_images_flat.view(b, t, c, h, w)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        optimizer.zero_grad()
        
        # ▼▼▼ 수정된 부분: 모델이 2개 항목 반환 ▼▼▼
        predicted_masks, reconstructed_images_flat = model(video_clip)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        
        # 1. 분할 손실 (Segmentation Loss)
        loss_focal = focal_loss(predicted_masks, ground_truth_masks)
        loss_dice = dice_loss(predicted_masks, ground_truth_masks)
        loss_seg = loss_focal + train_cfg['dice_weight'] * loss_dice

        # ▼▼▼ 수정된 부분: 2. 강화 손실 (Enhancement Loss) ▼▼▼
        # reconstructed_images_flat: (B*T, C, H, W), [0, 1] 범위
        # original_images_flat: (B*T, C, H, W), [0, 1] 범위
        loss_enhancement = l1_loss(reconstructed_images_flat, original_images_flat)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        # 3. 시간적 손실 (Temporal Loss)
        loss_temporal = 0
        if t > 1 and train_cfg['lambda_temporal'] > 0:
            img1_batch = video_clip[:, :-1].reshape(-1, c, h, w)
            img2_batch = video_clip[:, 1:].reshape(-1, c, h, w)
            if not torch.equal(img1_batch, img2_batch):
                img1_transformed, img2_transformed = raft_transforms(img1_batch, img2_batch)
                with torch.no_grad():
                    flows = raft_model(img1_transformed, img2_transformed)[-1]
                # flow 크기를 (h, w)로 리사이즈
                flows_resized = F.interpolate(flows, size=(h, w), mode='bilinear', align_corners=False) # <-- 'F' 사용 지점
                flows_unbatched = flows_resized.view(b, t - 1, 2, h, w)

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
        
        # ▼▼▼ 수정된 부분: 3개 손실을 조합 ▼▼▼
        total_loss = loss_seg + \
                     train_cfg['lambda_temporal'] * loss_temporal_avg + \
                     train_cfg['lambda_enhancement'] * loss_enhancement
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logging.error("!!! Loss is NaN or Inf. Stopping training.")
            return -1

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        current_loss = total_loss.item()
        epoch_loss += current_loss
        global_step = epoch * len(train_loader) + i
        
        # ▼▼▼ 수정된 부분: TensorBoard 로깅 세분화 ▼▼▼
        writer.add_scalar('Loss/Step/Total', current_loss, global_step)
        writer.add_scalar('Loss/Step/Segmentation', loss_seg.item(), global_step)
        writer.add_scalar('Loss/Step/Enhancement', loss_enhancement.item(), global_step)
        if t > 1 and train_cfg['lambda_temporal'] > 0:
            writer.add_scalar('Loss/Step/Temporal', loss_temporal_avg.item(), global_step)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    # ▼▼▼ 수정된 부분: 디버그 이미지 저장 (복원된 이미지 포함) ▼▼▼
    if (epoch + 1) % train_cfg['debug_image_interval'] == 0 and 'predicted_masks' in locals() and len(predicted_masks) > 0:
        pred_to_save = torch.sigmoid(predicted_masks[0, 0]).cpu()
        gt_to_save = ground_truth_masks[0, 0].cpu()
        
        recon_to_save = reconstructed_images_flat.view(b, t, c, h, w)[0, 0].cpu()
        orig_day_to_save = original_day_images[0, 0].cpu()
        orig_night_to_save = unnormalize(video_clip[0, 0]).cpu()

        debug_dir = os.path.join(train_cfg['log_dir'], 'debug_images')
        os.makedirs(debug_dir, exist_ok=True)
        
        save_image(pred_to_save, os.path.join(debug_dir, f'epoch_{epoch+1}_01_prediction.png'))
        save_image(gt_to_save, os.path.join(debug_dir, f'epoch_{epoch+1}_02_ground_truth.png'))
        save_image(recon_to_save, os.path.join(debug_dir, f'epoch_{epoch+1}_03_reconstructed_day.png'))
        save_image(orig_day_to_save, os.path.join(debug_dir, f'epoch_{epoch+1}_04_original_day_GT.png'))
        save_image(orig_night_to_save, os.path.join(debug_dir, f'epoch_{epoch+1}_05_original_night_Input.png'))
        
        writer.add_image('Images/Prediction', pred_to_save, epoch)
        writer.add_image('Images/GroundTruth', gt_to_save, epoch)
        writer.add_image('Images/Reconstructed_Day', recon_to_save, epoch)
        writer.add_image('Images/Original_Day_GT', orig_day_to_save, epoch)
        writer.add_image('Images/Original_Night_Input', orig_night_to_save, epoch)
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    return epoch_loss / len(train_loader) if len(train_loader) > 0 else 0

def validate(model, val_loader, device, focal_loss, dice_loss, l1_loss, train_cfg):
    model.eval()
    val_loss = 0
    val_seg_loss = 0
    val_enhance_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            if batch is None: continue
            
            # ▼▼▼ 수정된 부분: 3개 항목 수신 ▼▼▼
            video_clip, ground_truth_masks, original_day_images = batch
            video_clip = video_clip.to(device)
            ground_truth_masks = ground_truth_masks.to(device)
            original_day_images = original_day_images.to(device)
            
            b, t, c, h, w = video_clip.shape
            
            predicted_masks, reconstructed_images_flat = model(video_clip)
            
            # 1. 분할 손실
            loss_focal = focal_loss(predicted_masks, ground_truth_masks)
            loss_dice = dice_loss(predicted_masks, ground_truth_masks)
            loss_seg = loss_focal + train_cfg['dice_weight'] * loss_dice
            
            # 2. 강화 손실
            original_images_flat = original_day_images.view(b*t, c, h, w)
            loss_enhancement = l1_loss(reconstructed_images_flat, original_images_flat)
            
            # 3. 총 손실 (Validation에서는 Temporal Loss 제외)
            loss = loss_seg + train_cfg['lambda_enhancement'] * loss_enhancement
            
            val_loss += loss.item()
            val_seg_loss += loss_seg.item()
            val_enhance_loss += loss_enhancement.item()
            num_batches += 1
            
    if num_batches == 0:
        return 0, 0, 0
        
    avg_total_loss = val_loss / num_batches
    avg_seg_loss = val_seg_loss / num_batches
    avg_enhance_loss = val_enhance_loss / num_batches
    
    return avg_total_loss, avg_seg_loss, avg_enhance_loss
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

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
    logger.info("Model created.")

    if gpu_ids and len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg['lr'], weight_decay=train_cfg.get('weight_decay', 1e-5))
    logger.info(f"Optimizer: AdamW with lr={train_cfg['lr']}, weight_decay={train_cfg.get('weight_decay', 1e-5)}")
    
    if train_cfg['scheduler_name'] == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=train_cfg['T_0'], T_mult=train_cfg['T_mult'], eta_min=1e-6)
        logger.info(f"Using Scheduler: CosineAnnealingWarmRestarts")
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=train_cfg['scheduler_factor'], patience=train_cfg['scheduler_patience'])
        logger.info(f"Using Scheduler: ReduceLROnPlateau")
    
    raft_weights = Raft_Large_Weights.DEFAULT
    raft_transforms = raft_weights.transforms()
    raft_model = raft_large(weights=raft_weights, progress=False).to(device)
    raft_model.eval()

    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    l1_loss = nn.L1Loss()
    
    if train_cfg['dataset_type'] == 'folder':
        # ▼▼▼ 수정된 부분: original_data_root 전달 ▼▼▼
        if not train_cfg.get('original_data_root'):
             logger.error("config.py의 train 섹션에 'original_data_root' (원본 주간 이미지 경로) 설정이 필요합니다.")
             raise ValueError("'original_data_root' not set in config.py")
             
        train_dataset_full = FolderImageMaskDataset(
            root_dir=train_cfg['folder_data_root'], 
            original_data_root=train_cfg['original_data_root'],
            image_folder_name=train_cfg['image_folder_name'],
            mask_folder_name=train_cfg['mask_folder_name'], 
            clip_len=common_cfg['clip_len'], 
            is_train=True,
            use_augmentation=train_cfg.get('use_augmentation', True))
            
        val_dataset_full = FolderImageMaskDataset(
            root_dir=train_cfg['folder_data_root'], 
            original_data_root=train_cfg['original_data_root'],
            image_folder_name=train_cfg['image_folder_name'],
            mask_folder_name=train_cfg['mask_folder_name'], 
            clip_len=common_cfg['clip_len'], 
            is_train=False,
            use_augmentation=False)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
            
        logger.info(f"Loading folder-based dataset from: {train_cfg['folder_data_root']}")
        logger.info(f"Loading original day images from: {train_cfg['original_data_root']}")
        logger.info(f"Data Augmentation: {train_cfg.get('use_augmentation', True)}")
        
        total_size = len(train_dataset_full)
        train_size = int(0.8 * total_size)
        indices = list(range(total_size))
        random.shuffle(indices)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        train_dataset = Subset(train_dataset_full, train_indices)
        val_dataset = Subset(val_dataset_full, val_indices)
    else:
        raise ValueError(f"Invalid or unimplemented dataset_type: {train_cfg['dataset_type']}")

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if not batch: return None
        # ▼▼▼ 수정된 부분: 3개 항목 반환 가정 ▼▼▼
        try:
            return torch.utils.data.dataloader.default_collate(batch)
        except Exception as e:
            logging.warning(f"Collate function error, skipping batch: {e}")
            return None
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=common_cfg['num_workers'], collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=common_cfg['num_workers'], collate_fn=collate_fn)
    logger.info(f"Dataset split: {len(train_dataset)} for training, {len(val_dataset)} for validation.")
    
    # ▼▼▼ 수정된 부분: Subset 객체를 verify_and_save_samples에 전달 ▼▼▼
    verify_and_save_samples(train_dataset, common_cfg, train_cfg)
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    best_val_loss = np.inf
    early_stop_counter = 0

    logger.info("--- Starting Training (with Enhancement Loss) ---")
    for epoch in range(train_cfg['epochs']):
        train_loss = train_one_epoch(model, raft_model, raft_transforms, train_loader, optimizer, device, focal_loss, dice_loss, l1_loss, writer, epoch, train_cfg)
        
        if train_loss == -1: break
            
        # ▼▼▼ 수정된 부분: 3개의 val loss 수신 ▼▼▼
        val_loss, val_seg_loss, val_enhance_loss = validate(model, val_loader, device, focal_loss, dice_loss, l1_loss, train_cfg)
        
        logger.info(f"Epoch {epoch+1}/{train_cfg['epochs']} | Train Loss: {train_loss:.4f} | Val Total Loss: {val_loss:.4f} (Seg: {val_seg_loss:.4f}, Enhance: {val_enhance_loss:.4f})")
        
        writer.add_scalar('Loss/Epoch/Train_Total', train_loss, epoch)
        writer.add_scalar('Loss/Epoch/Val_Total', val_loss, epoch)
        writer.add_scalar('Loss/Epoch/Val_Segmentation', val_seg_loss, epoch)
        writer.add_scalar('Loss/Epoch/Val_Enhancement', val_enhance_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

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