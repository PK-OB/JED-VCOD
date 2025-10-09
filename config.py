class Config:
    # --- 공통 설정 ---
    common = {
        'gpu_ids': '0,1',
        'clip_len': 8,
        'num_workers': 4,
    }

    # --- 학습 설정 ---
    train = {
        'experiment_name': 'JED-VCOD_exp_final',
        'dataset_type': 'folder',
        
        'folder_data_root': '/home/sjy/paper/JED-VCOD/data/MoCA-Mask/Test_Night',
        'image_folder_name': 'Imgs',
        'mask_folder_name': 'GT',

        # ▼▼▼ 추가된 부분 ▼▼▼
        'use_augmentation': True, # 데이터 증강 활성화/비활성화 (True/False)
        
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs',
        'epochs': 1000,
        'batch_size': 16,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'lambda_temporal': 0.1,
        'dice_weight': 2.0,
        'patience': 100,
        'scheduler_patience': 5,
        'scheduler_factor': 0.5,
        'checkpoint_name': '1009_1502_aug.pth',
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