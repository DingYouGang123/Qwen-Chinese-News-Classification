import argparse
import json
import torch
from transformers import AutoTokenizer
from model_utils import load_qwen_model
from evaluate import predict_single_sample

def batch_infer(input_path, output_path, model, tokenizer):
    """批量推理（支持JSONL输入）"""
    results = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            messages = [
                {"role": "system", "content": "你是文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型"},
                {"role": "user", "content": f"文本：{data['text']}，类别选项：{data['category']}"}
            ]
            pred_label = predict_single_sample(messages, model, tokenizer)
            results.append({
                "text": data["text"],
                "category": data["category"],
                "predicted_output": pred_label
            })
    
    # 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(f"批量推理完成，结果保存到：{output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="中文新闻分类推理脚本")
    parser.add_argument("--model_path", type=str, default="./output/Qwen1.5-LoRA-adapter", help="模型路径（LoRA适配器）")
    parser.add_argument("--input_path", type=str, required=True, help="输入文件路径（JSONL格式，含text和category字段）")
    parser.add_argument("--output_path", type=str, default="infer_results.jsonl", help="输出文件路径")
    
    args = parser.parse_args()
    
    # 加载模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./qwen/Qwen1___5-7B-Chat/", use_fast=False, trust_remote_code=True)
    model = load_qwen_model(use_lora=True)
    model.load_adapter(args.model_path)  # 加载LoRA适配器
    model.eval()
    
    # 批量推理
    batch_infer(args.input_path, args.output_path, model, tokenizer)