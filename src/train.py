import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from torch.optim import AdamW

# Import các module nội bộ
from src.configs import Config
from src.dataset import HateSpeechDataset
from src.model import HateSpeechModel
from src.utils import calculate_metrics, seed_everything


def train_fn(data_loader, model, optimizer, device, scheduler, criterion):
    """
    Hàm huấn luyện cho 1 epoch (Dành cho PhoBERT)
    """
    model.train()
    total_loss = 0
    total_acc = 0
    total_f1 = 0

    # Tqdm progress bar
    tk0 = tqdm(data_loader, total=len(data_loader), desc="Training")

    for batch in tk0:
        # 1. Lấy dữ liệu từ batch (Dictionary do Dataset trả về)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 2. Xóa gradient cũ
        model.zero_grad()

        # 3. Forward Pass
        # outputs shape: [Batch_Size, N_Classes]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # 4. Tính Loss
        loss = criterion(outputs, labels)

        # 5. Backward Pass
        loss.backward()

        # 6. Cập nhật trọng số
        optimizer.step()
        scheduler.step()  # Cập nhật learning rate scheduler

        # 7. Tính chỉ số (Metrics) để theo dõi
        acc, f1 = calculate_metrics(outputs, labels)

        total_loss += loss.item()
        total_acc += acc
        total_f1 += f1

        # Hiển thị loss hiện tại trên thanh loading
        tk0.set_postfix(loss=loss.item())

    return total_loss / len(data_loader), total_acc / len(data_loader), total_f1 / len(data_loader)


def eval_fn(data_loader, model, device, criterion):
    """
    Hàm đánh giá model (Validation)
    """
    model.eval()  # Tắt Dropout
    total_loss = 0
    total_acc = 0
    total_f1 = 0

    with torch.no_grad():  # Không tính gradient
        tk0 = tqdm(data_loader, total=len(data_loader), desc="Validating")

        for batch in tk0:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = criterion(outputs, labels)

            acc, f1 = calculate_metrics(outputs, labels)

            total_loss += loss.item()
            total_acc += acc
            total_f1 += f1

    return total_loss / len(data_loader), total_acc / len(data_loader), total_f1 / len(data_loader)


def main():
    # 1. Khóa Seed để kết quả giống nhau 100% mỗi lần chạy
    seed_everything(42)

    # 2. Setup Device
    device = Config.DEVICE
    print(f"--> Đang chạy trên: {device} (Mode: PhoBERT Fine-tuning)")

    # 3. Load Dữ liệu (Sử dụng Dataset mới đã sửa ở bước trước)
    print("--> Đang chuẩn bị dữ liệu...")
    train_dataset = HateSpeechDataset(Config.TRAIN_PATH, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    dev_dataset = HateSpeechDataset(Config.DEV_PATH, is_train=False)
    dev_loader = DataLoader(dev_dataset, batch_size=Config.BATCH_SIZE)

    # 4. Khởi tạo Model
    print(f"--> Đang load model Pre-trained: {Config.MODEL_NAME}...")
    model = HateSpeechModel(n_classes=Config.N_CLASSES)
    model.to(device)

    # 5. Optimizer & Scheduler (Cực quan trọng cho Transformer)
    # Dùng AdamW thay vì Adam thường
    # Learning rate phải nhỏ (2e-5)
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Scheduler: Tăng dần LR lúc đầu (warmup) rồi giảm dần
    num_train_steps = int(len(train_loader) * Config.EPOCHS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    criterion = nn.CrossEntropyLoss().to(device)

    # 6. Training Loop
    best_valid_f1 = 0.0

    print("\nBẮT ĐẦU HUẤN LUYỆN...")
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")

        # Train
        train_loss, train_acc, train_f1 = train_fn(train_loader, model, optimizer, device, scheduler, criterion)

        # Validate
        valid_loss, valid_acc, valid_f1 = eval_fn(dev_loader, model, device, criterion)

        # In kết quả
        print(f"   Train Loss: {train_loss:.4f} | Train F1: {train_f1 * 100:.2f}%")
        print(f"   Val Loss  : {valid_loss:.4f} | Val F1  : {valid_f1 * 100:.2f}%")

        # Lưu Best Model theo F1
        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"--> Đã lưu model tốt nhất! (New Best F1: {best_valid_f1 * 100:.2f}%)")
        else:
            print(f"--> Model chưa cải thiện (Best F1: {best_valid_f1 * 100:.2f}%)")


if __name__ == "__main__":
    main()