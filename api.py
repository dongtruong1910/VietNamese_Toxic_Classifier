from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.predict import HateSpeechPredictor

app = FastAPI()
print("--> Đang khởi động Server...")
predictor = HateSpeechPredictor()


class Item(BaseModel):
    text: str


@app.post("/predict")
def predict(item: Item):
    # Gọi hàm predict thông minh (đã xử lý đoạn văn)
    result = predictor.predict(item.text)

    return {
        "text": item.text,
        "prediction": result['label'],
        "confidence": f"{result['confidence']:.2%}",
        "is_toxic": result['is_toxic'],
        "flagged_sentence": result['flagged_sentence']
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)