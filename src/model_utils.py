import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq

def load_qwen_model(model_path="./qwen/Qwen1___5-7B-Chat/", use_lora=True):
    """
    加载Qwen1.5-7B-Chat模型（支持LoRA适配）
    """
    # 基础模型配置
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.enable_input_require_grads()  # 开启梯度检查点支持
    
    if not use_lora:
        return model
    
    # LoRA配置（参数高效微调）
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        r=8,  # 低秩矩阵秩
        lora_alpha=32,  # 缩放因子
        lora_dropout=0.1,
        inference_mode=False
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 打印可训练参数占比（约0.1%）
    return model

def get_training_args(output_dir="./output", batch_size=4, epochs=2, lr=1e-4):
    """
    训练参数配置（模块化可调）
    """
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=epochs,
        save_steps=100,
        learning_rate=lr,
        save_on_each_node=True,
        gradient_checkpointing=True,  # 节省显存
        report_to="none",
        fp16=False,
        bf16=True  # 混合精度训练
    )

def build_trainer(model, tokenizer, train_dataset, training_args):
    """
    构建训练器
    """
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[]
    )

if __name__ == "__main__":
    # 测试模型加载
    model = load_qwen_model()
    print("模型加载完成，架构：", model.config.architectures)