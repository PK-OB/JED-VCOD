# config.py

class Config:
    # --- 공통 설정 ---
    common = {
        'gpu_ids': '0,1,2',
        'clip_len': 8,
        'num_workers': 4,
    }

    # --- 학습 설정 ---
    train = {
        'experiment_name': 'JED-VCOD_exp1',
        'data_root': 'data/Night-Camo-Fauna/',
        'annotation_file': 'data/MoCA/Annotations/annotations_modified.csv',
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs',
        'epochs': 10000,
        'batch_size': 8,
        'lr': 0.0001,
        'lambda_temporal': 0.1,
        'dice_weight': 2.0,
        'patience': 100,
        'scheduler_patience': 5,
        'scheduler_factor': 0.5,
        'checkpoint_name': '0922_1953_model.pth',
    }

    # --- 평가 설정 ---
    evaluate = {
        'experiment': 'proposed',
        'data_root': 'data/Night-Camo-Fauna/',
        'annotation_file': 'data/MoCA/Annotations/annotations_modified.csv',
        'checkpoint_path': 'checkpoints/best_model.pth',
        'batch_size': 1,
    }

cfg = Config()