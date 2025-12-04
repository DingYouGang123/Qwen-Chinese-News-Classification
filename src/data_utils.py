import json
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

def load_fudan_news_dataset(train_path=None, test_path=None, use_huggingface=False):
    """
    加载中文新闻分类数据集（支持本地JSONL或HuggingFace数据集）
    Args:
        train_path: 本地训练集路径
        test_path: 本地测试集路径
        use_huggingface: 是否使用HuggingFace公开数据集
    Returns:
        train_ds, test_ds: 处理后的Dataset对象
    """
    if use_huggingface:
        # 加载HuggingFace公开中文新闻数据集（备用方案）
        dataset = load_dataset("clue", "tnews")
        return dataset["train"], dataset["test"]
    
    # 加载本地JSONL数据集
    def _load_jsonl(path):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return pd.DataFrame(data)
    
    train_df = _load_jsonl(train_path) if train_path else None
    test_df = _load_jsonl(test_path) if test_path else None
    
    train_ds = Dataset.from_pandas(train_df) if train_df is not None else None
    test_ds = Dataset.from_pandas(test_df) if test_df is not None else None
    return train_ds, test_ds

def format_dataset_for_instruction(dataset, task_desc="文本分类"):
    """
    将数据集转换为「instruction-input-output」三元组格式
    """
    def _format_example(example):
        return {
            "instruction": f"你是{task_desc}专家，接收文本和候选类别，输出正确类别",
            "input": f"文本：{example['text']}，类别选项：{example['category']}",
            "output": example["output"]
        }
    
    return dataset.map(_format_example, remove_columns=dataset.column_names)

def process_func(example, tokenizer, max_length=384):
    """
    模型输入构建（含对话模板、损失掩码、截断）
    """
    # 系统指令+用户输入
    system_prompt = "<|im_start|>system\n你是文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型<|im_end|>\n"
    user_prompt = f"<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenize
    instruction_ids = tokenizer(system_prompt + user_prompt, add_special_tokens=False)
    response_ids = tokenizer(example["output"], add_special_tokens=False)
    
    # 拼接输入和标签（指令部分损失掩码为-100）
    input_ids = instruction_ids["input_ids"] + response_ids["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction_ids["attention_mask"] + response_ids["attention_mask"] + [1]
    labels = [-100] * len(instruction_ids["input_ids"]) + response_ids["input_ids"] + [tokenizer.pad_token_id]
    
    # 截断
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

if __name__ == "__main__":
    # 测试数据加载
    tokenizer = AutoTokenizer.from_pretrained("./qwen/Qwen1___5-7B-Chat/", use_fast=False, trust_remote_code=True)
    train_ds, test_ds = load_fudan_news_dataset(
        train_path="data/train.jsonl",
        test_path="data/test.jsonl"
    )
    formatted_train_ds = format_dataset_for_instruction(train_ds)
    processed_train_ds = formatted_train_ds.map(
        lambda x: process_func(x, tokenizer),
        remove_columns=formatted_train_ds.column_names
    )
    print("数据处理完成，样本示例：", processed_train_ds[0]["input_ids"][:10])