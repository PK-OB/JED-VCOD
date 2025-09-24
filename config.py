class Config:
    # --- 공통 설정 ---
    common = {
        'gpu_ids': '0,1,2',
        'clip_len': 8,
        'num_workers': 4,
    }

    # --- 학습 설정 ---
    train = {
        'experiment_name': 'JED-VCOD_exp_final',
        'data_root': 'data/Night-Camo-Fauna/',
        'annotation_file': '/home/sjy/paper/JED-VCOD/data/MoCA/Annotations/annotations.csv',
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs',
        'epochs': 1000,
        'batch_size': 8,
        'lr': 1e-4, # 안정적인 학습을 위해 0.0001로 설정
        'lambda_temporal': 0.1,
        'dice_weight': 2.0,
        'patience': 100,
        'scheduler_patience': 5,
        'scheduler_factor': 0.5,
        'checkpoint_name': '0922_2137_best_model.pth',
        'debug_image_interval': 5, # 5 에포크마다 예측 이미지 저장
    }

    # --- 평가 설정 ---
    evaluate = {
        'experiment': 'proposed',
        'data_root': 'data/Night-Camo-Fauna/',
        'annotation_file': 'data/MoCA/Annotations/annotations.csv',
        'checkpoint_path': 'checkpoints/0922_2137_best_model.pth',
        'batch_size': 1,
        'visualization_path': 'checkpoints/evaluate_result/evaluation_visualization.png' # 결과 이미지 저장 경로

    }

cfg = Config()