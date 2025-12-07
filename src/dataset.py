import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
from pyvi import ViTokenizer
from .configs import Config
import os


class HateSpeechDataset(Dataset):
    def __init__(self, data_path, is_train=True):
        # 1. Load Tokenizer xịn của VinAI
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

        # 2. Đọc và xử lý dữ liệu như cũ
        try:
            self.df = pd.read_csv(data_path)
        except:
            self.df = pd.read_csv(data_path, sep='\t')

        rename_map = {'body': 'text', 'content': 'text', 'free_text': 'text', 'label_id': 'label'}
        self.df.rename(columns=rename_map, inplace=True)
        self.df.dropna(subset=['text', 'label'], inplace=True)

        self.texts = self.df['text'].tolist()
        self.labels = self.df['label'].tolist()
        self.max_len = Config.MAX_LEN

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        # BƯỚC QUAN TRỌNG: Segment tiếng Việt trước (dùng PyVi)
        # PhoBERT thích chữ kiểu "Học_sinh" thay vì "Học sinh"
        text_segmented = ViTokenizer.tokenize(text)

        # Mã hóa bằng PhoBERT Tokenizer
        encoding = self.tokenizer.encode_plus(
            text_segmented,
            add_special_tokens=True,  # Thêm token đặc biệt [CLS], [SEP]
            max_length=self.max_len,
            padding='max_length',  # Đệm cho đủ chiều dài
            truncation=True,  # Cắt nếu dài quá
            return_attention_mask=True,  # Tạo mask để model không nhìn vào phần đệm
            return_tensors='pt',  # Trả về PyTorch Tensor
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }