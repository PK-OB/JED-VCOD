# pk-ob/jed-vcod/JED-VCOD-cc543b29cefb3a45b940bfd01f42c33af7a6bb25/config.py

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
        
        # ▼▼▼ 수정된 부분 ▼▼▼
        # 1. 야간(학습 입력) 데이터 경로
        'folder_data_root': '/home/sjy/paper/JED-VCOD/data/MoCA-Mask/Train_Night',
        
        # 2. 원본 주간(Enhancement 정답) 데이터 경로 (새로 추가!)
        # (Train_Night의 원본 경로로 수정해주세요)
        'original_data_root': '/home/sjy/paper/JED-VCOD/data/MoCA-Mask/Train', 
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        
        'image_folder_name': 'Imgs',
        'mask_folder_name': 'GT',
        
        # --- CutMix 설정 ---
        'use_cutmix': False,
        'cutmix_beta': 1.0,
        'cutmix_prob': 0.5,

        # --- Scheduler 설정 ---
        'scheduler_name': 'CosineAnnealingWarmRestarts', # 'ReduceLROnPlateau' 또는 'CosineAnnealingWarmRestarts'
        'T_0': 50,  # CosineAnnealingWarmRestarts: 첫 번째 재시작까지의 에포크 수
        'T_mult': 2, # CosineAnnealingWarmRestarts: 재시작 주기가 몇 배로 늘어날지
        
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs',
        'epochs': 5000, # 스케줄러에 맞춰 에포크 수 조절
        'batch_size': 16,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'lambda_temporal': 0.5,
        'dice_weight': 2.0,
        
        # ▼▼▼ 수정된 부분 ▼▼▼
        'lambda_enhancement': 0.5, # (새로운 Enhancement Loss 가중치)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        
        'patience': 100, # ReduceLROnPlateau: Early stopping patience
        'scheduler_patience': 5, # ReduceLROnPlateau: Scheduler patience
        'scheduler_factor': 0.5,
        'checkpoint_name': '1027_1620.pth',
        'debug_image_interval': 5,
    }
    
    # --- 평가 설정 ---
    evaluate = {
        'experiment': 'proposed',
        'batch_size': 4,
        'visualization_path': 'evaluation_results/new_test_set_visualization10.png', # (저장 경로 예시)
        'checkpoint_path': 'checkpoints/1026_2218.pth', #           (평가할 모델 경로)

        # ▼▼▼ 1. 여기에 새로운 설정을 추가합니다 ▼▼▼
        # -----------------------------------------------------------------
        'eval_dataset_type': 'folder',  # 'folder' 또는 'moca_csv' (기본값)
        
        # 'folder' 타입을 사용할 경우
        'eval_folder_data_root': '/home/sjy/paper/JED-VCOD/data/MoCA-Mask/Test_Night', # <-- 여기에 새 테스트셋 경로 입력!
        
        # ▼▼▼ 수정된 부분 ▼▼▼
        # (평가 시에도 원본 주간 이미지가 필요합니다. Test_Night의 원본 경로로 수정해주세요)
        'eval_original_data_root': '/home/sjy/paper/JED-VCOD/data/MoCA-Mask/Test',
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        
        'eval_image_folder_name': 'Imgs',
        'eval_mask_folder_name': 'GT',
        # -----------------------------------------------------------------
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        # 'moca_csv' 타입을 사용할 경우 (기존 설정)
        'data_root': 'data/Night-Camo-Fauna/',
        'annotation_file': 'data/MoCA/Annotations/annotations.csv',
    }

    # --- 박스 평가 설정 ---
    evaluate_box = {
        'annotation_file': 'data/MoCA/Annotations/annotations.csv',
        'checkpoint_path': 'checkpoints/best_model.pth',
        'iou_threshold': 0.5,
        'batch_size': 1,
    }

cfg = Config()