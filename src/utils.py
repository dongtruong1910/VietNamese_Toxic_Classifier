import os
import random

import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def seed_everything(seed=42):
    """
    Khóa mọi yếu tố ngẫu nhiên để kết quả có thể tái lập được.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_metrics(preds, labels):
    """
    Tính độ chính xác và F1-Score
    preds: Output của model (logits) -> chưa qua softmax
    labels: Nhãn thật
    """
    # Chuyển logits thành class ID (0, 1, 2)
    # Lấy vị trí có giá trị lớn nhất (argmax)
    pred_labels = torch.argmax(preds, dim=1).cpu().numpy()
    true_labels = labels.cpu().numpy()

    # Tính accuracy
    acc = accuracy_score(true_labels, pred_labels)

    # Tính F1-Score (Macro: trung bình cộng các lớp)
    # Quan trọng vì data Hate Speech thường bị lệch (Clean nhiều hơn Hate)
    f1 = f1_score(true_labels, pred_labels, average='macro')

    return acc, f1


def count_parameters(model):
    """Đếm xem model có bao nhiêu tham số (để báo cáo thầy)"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)