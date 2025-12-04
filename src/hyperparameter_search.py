import optuna
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
import swanlab

from data_utils import load_fudan_news_dataset, format_dataset_for_instruction, process_func
from model_utils import load_qwen_model, get_training_args, build_trainer
from evaluate import evaluate_model
from transformers import AutoTokenizer

def objective(trial):
    """Optuna超参数搜索目标函数"""
    # 搜索空间（基于统计优化思路设计）
    lora_r = trial.suggest_int("lora_r", 4, 16)  # LoRA秩
    lora_alpha = trial.suggest_int("lora_alpha", 16, 64)  # 缩放因子
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)  # 学习率（对数分布）
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])  # 批大小
    
    # 初始化实验
    swanlab.init(project="Qwen-Hyperparameter-Search", experiment_name=f"trial-{trial.number}")
    swanlab.log({"lora_r": lora_r, "lora_alpha": lora_alpha, "learning_rate": learning_rate, "batch_size": batch_size})
    
    # 加载tokenizer和模型（动态配置LoRA）
    tokenizer = AutoTokenizer.from_pretrained("./qwen/Qwen1___5-7B-Chat/", use_fast=False, trust_remote_code=True)
    model = load_qwen_model(use_lora=False)  # 先加载基础模型
    
    # 动态配置LoRA
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        inference_mode=False
    )
    model = get_peft_model(model, lora_config)
    
    # 加载数据集
    train_ds, test_ds = load_fudan_news_dataset(
        train_path="data/train.jsonl",
        test_path="data/test.jsonl"
    )
    formatted_train_ds = format_dataset_for_instruction(train_ds)
    formatted_test_ds = format_dataset_for_instruction(test_ds)
    processed_train_ds = formatted_train_ds.map(
        lambda x: process_func(x, tokenizer),
        remove_columns=formatted_train_ds.column_names
    )
    
    # 训练
    training_args = get_training_args(
        output_dir=f"./output/trial-{trial.number}",
        batch_size=batch_size,
        epochs=2,
        lr=learning_rate
    )
    trainer = build_trainer(model, tokenizer, processed_train_ds, training_args)
    trainer.train()
    
    # 评估（以F1分数为优化目标）
    model.eval()
    metrics = evaluate_model(model, tokenizer, formatted_test_ds)
    f1_score = metrics["f1_macro"]
    
    # 记录结果
    swanlab.log(metrics)
    swanlab.finish()
    
    return f1_score

if __name__ == "__main__":
    # 启动超参数搜索（最多10轮试验）
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    study.optimize(objective, n_trials=10)
    
    # 输出最优结果
    print("超参数搜索完成！")
    print(f"最优F1分数：{study.best_value:.4f}")
    print(f"最优超参数：{study.best_params}")
    
    # 保存最优参数
    with open("./results/best_hyperparams.json", "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, ensure_ascii=False, indent=2)