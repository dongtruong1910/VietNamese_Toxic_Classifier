import torch.nn as nn
from transformers import AutoModel
from .configs import Config


class HateSpeechModel(nn.Module):
    def __init__(self, n_classes):
        super(HateSpeechModel, self).__init__()
        # Load khung xương PhoBERT
        self.bert = AutoModel.from_pretrained(Config.MODEL_NAME, weights_only=False)

        # Khóa bớt các tầng đầu để train nhanh hơn (Optional - Tùy chọn)
        for param in self.bert.parameters():
            param.requires_grad = True

        # Thêm đầu ra phân loại
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(768, n_classes)  # 768 là kích thước vector của PhoBERT Base

    def forward(self, input_ids, attention_mask):
        # Cho dữ liệu chạy qua PhoBERT
        # output[0] là hidden states, output[1] là pooled output (vector đại diện câu)
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Lấy vector đại diện của token [CLS] (token đầu tiên)
        # Nó chứa ý nghĩa của toàn bộ câu
        pooled_output = outputs[1]

        output = self.drop(pooled_output)
        return self.fc(output)