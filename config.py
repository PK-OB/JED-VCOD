class Config:
    # --- 공통 설정 ---
    common = {
        'gpu_ids': '0,1',
        'clip_len': 8,
        'num_workers': 4,
    }

    # --- 학습 설정 ---
    train = {
        'experiment_name': 'JED-VCOD_exp_advanced',
        'dataset_type': 'folder',
        
        'folder_data_root': '/home/sjy/paper/JED-VCOD/data/MoCA-Mask/Train_Night',
        'image_folder_name': 'Imgs',
        'mask_folder_name': 'GT',
        
        # --- CutMix 설정 ---
        'use_cutmix': True,
        'cutmix_beta': 1.0,
        'cutmix_prob': 0.5,

        # --- Scheduler 설정 ---
        'scheduler_name': 'CosineAnnealingWarmRestarts', # 'ReduceLROnPlateau' 또는 'CosineAnnealingWarmRestarts'
        'T_0': 10,  # CosineAnnealingWarmRestarts: 첫 번째 재시작까지의 에포크 수
        'T_mult': 2, # CosineAnnealingWarmRestarts: 재시작 주기가 몇 배로 늘어날지
        
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs',
        'epochs': 1000, # 스케줄러에 맞춰 에포크 수 조절
        'batch_size': 16,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'lambda_temporal': 0.1,
        'dice_weight': 2.0,
        'patience': 100, # ReduceLROnPlateau: Early stopping patience
        'scheduler_patience': 5, # ReduceLROnPlateau: Scheduler patience
        'scheduler_factor': 0.5,
        'checkpoint_name': '1009_1603_cutmix.pth',
        'debug_image_interval': 5,
    }
    
    # --- 평가 설정 ---
    evaluate = {
        'experiment': 'proposed',
        'data_root': 'data/Night-Camo-Fauna/',
        'annotation_file': 'data/MoCA/Annotations/annotations.csv',
        'checkpoint_path': 'checkpoints/best_model.pth',
        'batch_size': 1,
        'visualization_path': 'evaluation_visualization.png'
    }

    # --- 박스 평가 설정 ---
    evaluate_box = {
        'annotation_file': 'data/MoCA/Annotations/annotations.csv',
        'checkpoint_path': 'checkpoints/best_model.pth',
        'iou_threshold': 0.5,
        'batch_size': 1,
    }

cfg = Config()