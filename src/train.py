import swanlab
from swanlab.integration.transformers import SwanLabCallback
from transformers import AutoTokenizer

from data_utils import load_fudan_news_dataset, format_dataset_for_instruction, process_func
from model_utils import load_qwen_model, get_training_args, build_trainer
from evaluate import evaluate_model

if __name__ == "__main__":
    # 初始化实验监控
    swanlab.init(project="Qwen-Chinese-News-Classification", experiment_name="Qwen1.5-7B-LoRA")
    
    # 1. 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained("./qwen/Qwen1___5-7B-Chat/", use_fast=False, trust_remote_code=True)
    model = load_qwen_model(use_lora=True)
    
    # 2. 加载并处理数据集
    train_ds, test_ds = load_fudan_news_dataset(
        train_path="data/train.jsonl",
        test_path="data/test.jsonl"
    )
    formatted_train_ds = format_dataset_for_instruction(train_ds)
    formatted_test_ds = format_dataset_for_instruction(test_ds)
    
    # 预处理（适配模型输入）
    processed_train_ds = formatted_train_ds.map(
        lambda x: process_func(x, tokenizer, max_length=384),
        remove_columns=formatted_train_ds.column_names
    )
    
    # 3. 配置训练参数并启动训练
    training_args = get_training_args(
        output_dir="./output/Qwen1.5-LoRA",
        batch_size=4,
        epochs=2,
        lr=1e-4
    )
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=processed_train_ds,
        training_args=training_args
    )
    trainer.add_callback(SwanLabCallback())
    trainer.train()
    
    # 4. 模型评估
    model.eval()
    metrics = evaluate_model(model, tokenizer, formatted_test_ds)
    print("最终评估结果：", metrics)
    
    # 5. 保存模型（仅保存LoRA适配器，节省空间）
    model.save_pretrained("./output/Qwen1.5-LoRA-adapter")
    swanlab.finish()