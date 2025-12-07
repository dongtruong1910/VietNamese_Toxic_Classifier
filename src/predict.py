import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from pyvi import ViTokenizer
from src.configs import Config
from src.model import HateSpeechModel
import os
import re


class HateSpeechPredictor:
    def __init__(self, model_path=None):
        self.device = Config.DEVICE
        print(f"--> Đang khởi tạo Predictor trên: {self.device}")

        # 1. Load Tokenizer & Model
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.model = HateSpeechModel(n_classes=Config.N_CLASSES)

        # 2. Load Weights
        if model_path is None:
            model_path = Config.MODEL_SAVE_PATH

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print("--> Đã load model thành công!")
        else:
            raise FileNotFoundError(f"Chưa có file model tại {model_path}")

        # Map nhãn
        self.labels_map = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}
        # Map mức độ nghiêm trọng (để so sánh)
        self.severity_map = {"CLEAN": 0, "OFFENSIVE": 1, "HATE": 2}

    def _split_sentences(self, text):
        """Hàm tách đoạn văn thành các câu nhỏ"""
        # Tách dựa trên dấu chấm, chấm than, chấm hỏi, hoặc xuống dòng
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|\n)\s', text)
        return [s.strip() for s in sentences if len(s.strip()) > 1]

    def _predict_single(self, text):
        """Dự đoán cho 1 câu đơn"""
        text_segmented = ViTokenizer.tokenize(text)

        encoding = self.tokenizer.encode_plus(
            text_segmented,
            max_length=Config.MAX_LEN,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probs = F.softmax(outputs, dim=1)

        max_prob, pred_idx = torch.max(probs, dim=1)
        return self.labels_map[pred_idx.item()], max_prob.item()

    def predict(self, text):
        """
        Hàm chính: Xử lý cả đoạn văn.
        Logic: Tách câu -> Dự đoán từng câu -> Lấy nhãn NẶNG NHẤT.
        """
        sentences = self._split_sentences(text)

        final_label = "CLEAN"
        final_conf = 0.0
        max_severity = 0
        flagged_sentence = ""  # Lưu lại câu bị vi phạm

        # Nếu đoạn văn quá ngắn hoặc không tách được, coi là 1 câu
        if len(sentences) == 0:
            sentences = [text]

        for sent in sentences:
            label, conf = self._predict_single(sent)
            severity = self.severity_map[label]

            # Cập nhật nếu tìm thấy câu nặng hơn (HATE > OFFENSIVE > CLEAN)
            # Hoặc cùng mức độ nhưng độ tin cậy cao hơn
            if severity > max_severity:
                max_severity = severity
                final_label = label
                final_conf = conf
                flagged_sentence = sent
            elif severity == max_severity and conf > final_conf:
                final_conf = conf
                flagged_sentence = sent

        # Nếu là CLEAN thì không cần flagged_sentence
        if final_label == "CLEAN":
            flagged_sentence = None

        return {
            "label": final_label,
            "confidence": final_conf,
            "is_toxic": final_label != "CLEAN",
            "flagged_sentence": flagged_sentence  # Câu "tội đồ" làm bài bị chặn
        }


# Test
if __name__ == "__main__":
    p = HateSpeechPredictor()
    # Test đoạn văn dài
    paragraph = "Hôm nay trời đẹp. Nhưng mày là đồ ngu. Đi chơi thôi."
    result = p.predict(paragraph)

    print(f"Input: {paragraph}")
    print(f"Kết quả: {result['label']} ({result['confidence']:.2%})")
    print(f"Câu vi phạm: {result['flagged_sentence']}")