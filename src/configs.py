import torch
import os

# Lấy đường dẫn gốc của dự án
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config:
    # --- ĐƯỜNG DẪN DỮ LIỆU ---
    TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'train.csv')
    DEV_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'dev.csv')
    TEST_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'test.csv')

    # Nơi lưu model
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'saved_models', 'best_model.pth')

    # --- CẤU HÌNH PHOBERT ---
    MODEL_NAME = "vinai/phobert-base"

    # Tham số xử lý văn bản
    MAX_LEN = 100  # Độ dài câu tối đa
    N_CLASSES = 3  # <--- DÒNG BẠN ĐANG THIẾU (0: Clean, 1: Offensive, 2: Hate)

    # --- THAM SỐ HUẤN LUYỆN (Fine-tuning) ---
    BATCH_SIZE = 16 # PhoBERT nặng nên để batch size nhỏ (16 hoặc 8)
    EPOCHS = 10
    LEARNING_RATE = 2e-5  # Learning rate rất nhỏ cho Transformer

    # Tự động chọn GPU
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    print(f"Device: {Config.DEVICE}")