ViHSD-BiLSTM-Attention/
│
├── data/
│   ├── raw/                # train.csv, dev.csv, test.csv
│   └── processed/          # vocab.pkl (Lưu từ điển)
│
├── src/                    # [CODEBASE CHÍNH]
│   ├── __init__.py         # Đánh dấu đây là package
│   ├── configs.py          # Cấu hình tham số
│   ├── dataset.py          # Xử lý dữ liệu
│   ├── model.py            # Kiến trúc Bi-LSTM + Attention
│   ├── utils.py            # Hàm phụ trợ
│   ├── train.py            # File chạy huấn luyện
│   └── predict.py          # File chạy dự đoán
│
├── requirements.txt
└── README.md