from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from transformers import AutoTokenizer
from model_utils import load_qwen_model
from evaluate import predict_single_sample

# 初始化API
app = FastAPI(title="中文新闻分类API", description="基于Qwen1.5-7B-LoRA的文本分类服务")

# 加载模型（启动时加载，避免重复初始化）
tokenizer = AutoTokenizer.from_pretrained("./qwen/Qwen1___5-7B-Chat/", use_fast=False, trust_remote_code=True)
model = load_qwen_model(use_lora=True)
model.load_adapter("./output/Qwen1.5-LoRA-adapter")
model.eval()

# 定义请求体格式
class ClassifyRequest(BaseModel):
    text: str  # 待分类文本
    category: str  # 候选类别（用逗号分隔，如"财经,科技,体育"）

# 定义响应体格式
class ClassifyResponse(BaseModel):
    predicted_category: str  # 预测类别
    status: str = "success"

@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest):
    try:
        messages = [
            {"role": "system", "content": "你是文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型"},
            {"role": "user", "content": f"文本：{request.text}，类别选项：{request.category}"}
        ]
        pred_label = predict_single_sample(messages, model, tokenizer)
        return ClassifyResponse(predicted_category=pred_label)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败：{str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "Qwen1.5-7B-LoRA"}

if __name__ == "__main__":
    # 启动服务（默认端口8000）
    uvicorn.run(app, host="0.0.0.0", port=8000)