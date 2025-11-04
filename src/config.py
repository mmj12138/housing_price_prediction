import os

DATA_DIR = os.path.join('data', 'raw')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')

CONFIG = {
    'target': 'SalePrice',
    'val_size': 0.3,
    'random_state': 42,
    'mlp_params': {
        'hidden_dims': [128, 64, 32],
        'dropout': 0.2,
        'lr': 1e-3,
        'epochs': 50,
        'batch_size': 64
    }
}
