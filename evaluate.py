import torch
import os
import argparse
import numpy as np
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torch.utils.data import DataLoader
from torch.nn.functional import grid_sample
from tqdm import tqdm

from models.eval_models import JED_VCOD_Fauna_Simplified_Eval, YourSOTAVCODModel, YourSOTAEnhancerModel
from datasets.moca_video_dataset import MoCAVideoDataset

def calculate_metrics(pred, gt):
    """NumPy 배열을 입력받아 MAE와 F-beta score를 계산하는 함수"""
    mae = np.mean(np.abs(pred - gt))
    pred_binary = (pred > 0.5).astype(np.uint8)
    gt_binary = (gt > 0.5).astype(np.uint8)
    epsilon = 1e-6
    tp = np.sum(pred_binary & gt_binary)
    fp = np.sum(pred_binary & ~gt_binary)
    fn = np.sum(~pred_binary & gt_binary)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    beta_sq = 0.3
    f_beta = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall + epsilon)
    return mae, f_beta

def main(args):
    # 1. 초기화 및 GPU 설정
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Evaluation ---")
    print(f"Experiment Type: {args.experiment}")
    print(f"Using device: {device}")

    # 2. 실험 종류에 따른 모델 구조 선택
    enhancer_model = None
    if args.experiment == 'baseline_b':
        from models.eval_models import YourSOTAEnhancerModel
        if not args.enhancer_checkpoint_path:
            raise ValueError("Baseline B requires --enhancer_checkpoint_path")
        enhancer_model = YourSOTAEnhancerModel().to(device)
        enhancer_model.load_state_dict(torch.load(args.enhancer_checkpoint_path, map_location=device))
        enhancer_model.eval()
        print(f"Loaded Enhancer model from: {args.enhancer_checkpoint_path}")

    if args.experiment in ['proposed', 'ablation_1']:
        model = JED_VCOD_Fauna_Simplified_Eval(use_dae=True)
    elif args.experiment == 'ablation_2':
        model = JED_VCOD_Fauna_Simplified_Eval(use_dae=False)
    elif args.experiment in ['baseline_a', 'baseline_b']:
        from models.eval_models import YourSOTAVCODModel
        model = YourSOTAVCODModel()
    else:
        raise ValueError(f"Unknown experiment: {args.experiment}")

    state_dict = torch.load(args.checkpoint_path, map_location=device)
    if all(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded Detection model from: {args.checkpoint_path}")

    # 3. RAFT 모델 및 데이터 로더 설정
    raft_weights = Raft_Large_Weights.DEFAULT
    raft_transforms = raft_weights.transforms()
    raft_model = raft_large(weights=raft_weights, progress=False).to(device)
    raft_model.eval()
    l1_loss = torch.nn.L1Loss()

    # ▼▼▼ 수정된 부분 ▼▼▼
    # MoCAVideoDataset 생성 시 synthetic_data_root 인자를 전달합니다.
    test_dataset = MoCAVideoDataset(
        synthetic_data_root=args.data_root, 
        annotation_file=args.annotation_file, 
        clip_len=args.clip_len
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 4. 평가 지표 변수 초기화
    total_mae, total_fbeta = 0.0, 0.0
    total_warping_error = 0.0
    mask_count = 0
    temporal_comparison_count = 0

    # 5. 평가 루프
    with torch.no_grad():
        for video_clip, ground_truth_masks in tqdm(test_loader, desc="Evaluating"):
            video_clip = video_clip.to(device)
            ground_truth_masks = ground_truth_masks.to(device)

            video_for_detection = video_clip
            if enhancer_model:
                b, t, c, h, w = video_clip.shape
                enhancer_input = video_clip.view(b * t, c, h, w)
                enhanced_frames = enhancer_model(enhancer_input)
                video_for_detection = enhanced_frames.view(b, t, c, h, w)
            
            predicted_logits = model(video_for_detection)
            predicted_masks = torch.sigmoid(predicted_logits)

            b, t, c, h, w = predicted_masks.shape

            for i in range(b):
                for j in range(t):
                    pred_mask_np = predicted_masks[i, j].squeeze().cpu().numpy()
                    gt_mask_np = ground_truth_masks[i, j].squeeze().cpu().numpy()
                    mae, fbeta = calculate_metrics(pred_mask_np, gt_mask_np)
                    total_mae += mae
                    total_fbeta += fbeta
                    mask_count += 1
            
            if t > 1:
                img1_batch = video_clip[:, :-1].reshape(-1, 3, h, w)
                img2_batch = video_clip[:, 1:].reshape(-1, 3, h, w)
                img1_transformed, img2_transformed = raft_transforms(img1_batch, img2_batch)
                flows = raft_model(img1_transformed, img2_transformed)[-1]
                flows_unbatched = flows.view(b, t - 1, 2, h, w)

                for i in range(t - 1):
                    flow_i = flows_unbatched[:, i]
                    grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
                    grid = torch.stack((grid_x, grid_y), 2).float()
                    displacement = flow_i.permute(0, 2, 3, 1)
                    warped_grid = grid + displacement
                    warped_grid[..., 0] = 2.0 * warped_grid[..., 0] / (w - 1) - 1.0
                    warped_grid[..., 1] = 2.0 * warped_grid[..., 1] / (h - 1) - 1.0
                    
                    mask_t = predicted_masks[:, i]
                    mask_t_plus_1 = predicted_masks[:, i + 1]
                    mask_t_warped = grid_sample(mask_t, warped_grid, mode='bilinear', padding_mode='border', align_corners=False)
                    
                    total_warping_error += l1_loss(mask_t_warped, mask_t_plus_1).item()
                    temporal_comparison_count += b

    # 6. 최종 결과 출력
    avg_mae = total_mae / mask_count if mask_count > 0 else 0
    avg_fbeta = total_fbeta / mask_count if mask_count > 0 else 0
    avg_warping_error = total_warping_error / temporal_comparison_count if temporal_comparison_count > 0 else 0

    print("\n--- Evaluation Results ---")
    print(f"Mean Absolute Error (MAE): {avg_mae:.4f}")
    print(f"F-beta Score (F_beta):    {avg_fbeta:.4f}")
    print(f"Warping Error:            {avg_warping_error:.4f}")
    print("--------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="JED-VCOD-Fauna Unified Evaluation Script")
    
    parser.add_argument('--experiment', type=str, required=True, 
                        choices=['proposed', 'baseline_a', 'baseline_b', 'ablation_1', 'ablation_2'],
                        help='평가할 실험 종류 선택')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='평가할 메인 모델의 .pth 파일 경로')
    parser.add_argument('--enhancer_checkpoint_path', type=str, default=None, help='(베이스라인 B 전용) Enhancer 모델의 .pth 파일 경로')
    parser.add_argument('--annotation_file', type=str, required=True, help='테스트 어노테이션 파일 경로')
    parser.add_argument('--gpu_id', type=str, default='0', help='사용할 GPU ID')
    
    # data_root 인자를 추가했습니다.
    parser.add_argument('--data_root', type=str, default='data/Night-Camo-Fauna/', help='테스트 데이터셋의 루트 경로')
    
    parser.add_argument('--batch_size', type=int, default=1, help='배치 크기')
    parser.add_argument('--clip_len', type=int, default=8, help='비디오 클립의 길이 (프레임 수)')
    parser.add_argument('--num_workers', type=int, default=0, help='데이터 로딩에 사용할 워커 수')

    args = parser.parse_args()
    main(args)