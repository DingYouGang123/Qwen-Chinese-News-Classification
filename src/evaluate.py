import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import swanlab
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def clean_prediction(pred):
    """清理模型输出（去除冗余文本）"""
    pred = pred.strip()
    pred = pred.split('\n')[0].split('。')[0].split('，')[0]
    return pred

def predict_single_sample(messages, model, tokenizer):
    """单样本推理"""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return clean_prediction(response)

def evaluate_model(model, tokenizer, test_dataset, save_results=True):
    """
    多指标评估（准确率、精确率、召回率、F1、混淆矩阵）
    """
    true_labels = []
    pred_labels = []
    test_text_list = []
    
    # 推理所有测试样本
    for idx, example in enumerate(test_dataset):
        messages = [
            {"role": "system", "content": example["instruction"]},
            {"role": "user", "content": example["input"]}
        ]
        true_label = example["output"]
        pred_label = predict_single_sample(messages, model, tokenizer)
        
        true_labels.append(true_label)
        pred_labels.append(pred_label)
        
        # 记录详细结果
        result_text = f"真实标签：{true_label}\n预测标签：{pred_label}\n正确：{true_label == pred_label}\n输入：{example['input']}"
        test_text_list.append(swanlab.Text(result_text, caption=f"样本{idx}"))
    
    # 计算多指标（macro平均，适配多分类）
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="macro", zero_division=0
    )
    
    # 生成混淆矩阵
    unique_labels = list(set(true_labels + pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.title("中文新闻分类混淆矩阵")
    plt.tight_layout()
    plt.savefig("./results/confusion_matrix.png")
    
    # 输出评估报告
    metrics = {
        "accuracy": round(accuracy, 4),
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
        "f1_macro": round(f1, 4)
    }
    
    if save_results:
        # 保存指标到SwanLab
        swanlab.log({**metrics, "预测详情": test_text_list})
        # 保存评估报告到本地
        with open("./results/evaluation_report.txt", "w", encoding="utf-8") as f:
            f.write(f"评估样本数：{len(true_labels)}\n")
            f.write(f"准确率：{accuracy:.4f}\n")
            f.write(f"宏精确率：{precision:.4f}\n")
            f.write(f"宏召回率：{recall:.4f}\n")
            f.write(f"宏F1分数：{f1:.4f}\n")
    
    return metrics

if __name__ == "__main__":
    from model_utils import load_qwen_model
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("./qwen/Qwen1___5-7B-Chat/", use_fast=False, trust_remote_code=True)
    model = load_qwen_model()
    model.eval()
    
    # 加载测试集
    from data_utils import load_fudan_news_dataset, format_dataset_for_instruction
    _, test_ds = load_fudan_news_dataset(test_path="data/test.jsonl")
    formatted_test_ds = format_dataset_for_instruction(test_ds)
    
    # 评估
    metrics = evaluate_model(model, tokenizer, formatted_test_ds)
    print("评估完成：", metrics)