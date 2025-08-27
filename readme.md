<p align="center">

<img src="./DeepSpeed-SFT/assets/image/ds-shiba.png" alt="DeepSpeed Shiba Inu!"/>

</p>

<div align="center">

## DeepSpeed 分布式微调实战：大模型加速训练方案

</div>

### 🚀 介绍
本项目基于 LLaMA3.1-8B-Chinese 开源基座模型，所用的GPU是4块NVIDIA RTX A6000，聚焦医学问答任务，设计并实现了一套大模型 领域化微调与部署流程。通过结合DeepSpeed分布式训练、LoRA 微调、量化压缩、RAG 检索增强 等前沿技术，显著提升模型在专业医学问题上的回答准确率与专业度，同时大幅降低推理成本。因为这个是我在学习大模型时的项目，可能有一些问题，欢迎提issue，希望能对大家学习大模型有帮助。现在只有基于DeepSpeed分布式训练的代码，其他代码正在整理中。

### 🐼 安装依赖
```bash
git clone https://github.com/RyanZxucheng/deepspeed-sft.git
cd DeepSpeed-SFT
pip install -r requirements.txt
```
如果服务器上无法连接huggingfacece，可以用清华镜像源
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

###  启动训练
```bash
cd training\supervised_finetuning\training_scripts\other_language
bash run_chinese.sh ./output  #./output这个参数可以自己的换成输出目录
```

###  如何使用自己的数据集
除了示例脚本中使用的数据集之外，您还可以添加和使用自己的数据集。为此，首先需要在dschat/utils/data/raw_datasets.py中添加一个新类来定义使用数据时的格式。需要确保遵循在PromptRawDataset类中定义的api和格式，以确保DeepSpeed所依赖的数据格式一致。可以查看现有的类来了解如何做。

其次，需要在dschat/utils/data/data_utils.py中的get_raw_dataset函数中添加与新数据集对应的if条件。if条件中的dataset_name字符串应该是您将作为训练脚本参数提供的数据集名称。最后，需要修改训练脚本中的“--data_path”参数中。

###  测试

```
export CUDA_VISIBLE_DEVICES=0
python prompt_eval.py \
--model_name_or_path_baseline XXX \
--model_name_or_path_finetune XXX
```
分别改成你的基座模型和微调模型的路径
