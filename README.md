# Qwen-Chinese-News-Classification
基于Qwen1.5-7B-Chat与LoRA的中文新闻分类系统，实现**参数高效微调、多指标评估、工程化部署**的端到端解决方案。

## 项目亮点
1. **参数高效微调**：采用LoRA技术，仅训练0.1%原始参数（约700万），单卡RTX4090 26.4分钟完成训练，显存占用降低60%。
2. **生成式模型适配判别任务**：设计「instruction-input-output」三元组模板+损失掩码机制，解决LLM在分类任务中的对齐问题。
3. **多维度评估体系**：支持准确率、精确率、召回率、F1-score（宏平均）与混淆矩阵可视化，符合统计分析标准。
4. **工程化落地能力**：提供批量推理脚本与FastAPI部署接口，支持工业级场景调用。
5. **统计优化支持**：集成Optuna超参数搜索，基于对数分布、离散采样等统计方法优化模型性能。

## 技术栈
- 大语言模型：Qwen1.5-7B-Chat（Decoder-only Transformer）
- 微调技术：LoRA（参数高效微调）
- 深度学习框架：PyTorch、Transformers、PEFT
- 数据处理：Pandas、Hugging Face Datasets
- 评估工具：Scikit-learn、Seaborn、SwanLab
- 工程部署：FastAPI、Uvicorn
- 超参数优化：Optuna

## 数据集
采用「复旦大学中文新闻分类数据集（zh_cls_fudan-news）」：
- 规模：20,000篇新闻，20个类别（财经、科技、体育等）
- 文本长度：单篇500-800字符（标题+正文）
- 标签准确率：≥98%（无明显噪声）

> 数据集获取：请参考[data/README.md](data/README.md)

## 快速开始
### 1. 环境配置
```bash
# 克隆仓库
git clone https://github.com/DingYouGang123/Qwen-Chinese-News-Classification.git
cd Qwen-Chinese-News-Classification

# 安装依赖
pip install -r requirements.txt
```

### 2. 模型下载
下载 Qwen1.5-7B-Chat 模型到本地目录：
```bash
# 从ModelScope下载
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('qwen/Qwen1.5-7B-Chat', cache_dir='./qwen', revision='master')"
```

### 3. 训练模型
```bash
# 运行主训练脚本
python src/train.py
```
训练日志实时同步到 SwanLab
模型输出：./output/Qwen1.5-LoRA-adapter（仅 LoRA 适配器）

### 4. 批量推理
```bash
# 批量处理JSONL格式输入
python src/infer.py --input_path ./data/test.jsonl --output_path ./infer_results.jsonl
```

### 5. 启动API服务
```bash
# 启动FastAPI服务
python src/deploy.py
# 访问 http://localhost:8000/docs 查看API文档
```

## 项目结构说明
```plaintext
src/
├── data_utils.py          # 数据集加载、格式转换、模型输入构建
├── model_utils.py         # Qwen模型加载、LoRA配置、训练器构建
├── evaluate.py            # 多指标评估、混淆矩阵生成、结果保存
├── train.py               # 端到端训练流程（数据→训练→评估）
├── infer.py               # 批量推理工具
├── deploy.py              # FastAPI部署接口
└── hyperparameter_search.py # Optuna超参数搜索
```

## 扩展方向
1. 支持长文本分类：引入文本分段 + 注意力聚合机制，利用 Qwen32K 上下文窗口优势。
2. 多语言迁移：适配英文、日文新闻数据集，验证跨语言分类能力。
3. 持续学习：实现新类别增量训练，无需重新微调整个模型。
4. 不确定性量化：输出分类置信度，支持高风险场景（如金融舆情）的安全调用。

## 联系作者
如有问题或建议，欢迎提交 Issue 或联系：210810518@stu.hit.edu.cn