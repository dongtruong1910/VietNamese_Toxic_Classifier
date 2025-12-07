import os

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from transformers import AutoTokenizer
from pyvi import ViTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

# Import nội bộ
from src.configs import Config
from src.model import HateSpeechModel


# --- 1. Tạo Dataset riêng cho việc Test (để không bị dính logic chia train/dev cũ) ---
class TestDataset(Dataset):
    def __init__(self, data_path):
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        # Đọc file test
        try:
            self.df = pd.read_csv(data_path)
        except:
            self.df = pd.read_csv(data_path, sep='\t')

        rename_map = {'body': 'text', 'content': 'text', 'free_text': 'text', 'label_id': 'label'}
        self.df.rename(columns=rename_map, inplace=True)
        self.df.dropna(subset=['text', 'label'], inplace=True)

        self.texts = self.df['text'].tolist()
        self.labels = self.df['label'].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        # Tách từ bằng PyVi trước khi đưa vào PhoBERT
        text_segmented = ViTokenizer.tokenize(text)

        encoding = self.tokenizer.encode_plus(
            text_segmented,
            add_special_tokens=True,
            max_length=Config.MAX_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(int(self.labels[idx]), dtype=torch.long)
        }


# --- 2. Hàm đánh giá chính ---
def evaluate_test():
    device = Config.DEVICE
    print(f"--> Đang đánh giá trên thiết bị: {device}")

    # Load Test Data
    test_dataset = TestDataset(Config.TEST_PATH)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    print(f"--> Số lượng mẫu kiểm tra: {len(test_dataset)}")

    # Load Model
    model = HateSpeechModel(n_classes=Config.N_CLASSES)
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    # Chạy dự đoán
    print("--> Đang chạy model trên tập Test...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- 3. Tính toán chỉ số ---
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')

    print("\n" + "=" * 30)
    print(f"KẾT QUẢ CUỐI CÙNG (TEST SET)")
    print("=" * 30)
    print(f"Accuracy : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"F1-Macro : {f1_macro:.4f} ({f1_macro * 100:.2f}%)")
    print("-" * 30)

    # In báo cáo chi tiết từng lớp
    target_names = ['Clean (0)', 'Offensive (1)', 'Hate (2)']
    print(classification_report(all_labels, all_preds, target_names=target_names))

    # --- 4. Vẽ biểu đồ Ma trận nhầm lẫn (Confusion Matrix) ---
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.ylabel('Nhãn thực tế (Actual)')
    plt.xlabel('Nhãn dự đoán (Predicted)')
    plt.title('Ma trận nhầm lẫn (Confusion Matrix)')
    plt.tight_layout()
    # Tạo đường dẫn lưu file
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'confusion_matrix.png')

    plt.savefig(save_path, dpi=300)  # dpi=300 để ảnh nét căng
    print(f"--> Đã lưu biểu đồ tại: {save_path}")


if __name__ == "__main__":
    evaluate_test()