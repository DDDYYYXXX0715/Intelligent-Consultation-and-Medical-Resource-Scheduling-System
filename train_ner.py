import os
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments

# ================= 1. 配置参数 =================
DATA_FILE = "D:\\code\\C++\\Python\\智能问诊与医疗资源调度系统\\train_bio.txt"  
MODEL_NAME = "bert-base-chinese" 
MAX_LEN = 128 

LABEL_LIST = ["O", "B-Symptom", "I-Symptom", "B-Disease", "I-Disease"]
LABEL_MAP = {label: i for i, label in enumerate(LABEL_LIST)}

# ================= 2. 数据处理 (加入废话过滤逻辑) =================
class MedicalNERDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts, self.labels = [], []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if not line:
                    if words:
                        # 【修改点 1】过滤掉全都是 'O' 的废话句子
                        if any(tag != 0 for tag in tags):
                            self.texts.append(words)
                            self.labels.append(tags)
                        words, tags = [], []
                else:
                    parts = line.split(" ")
                    if len(parts) >= 2:
                        words.append(parts[0])
                        tags.append(LABEL_MAP.get(parts[-1], 0)) # 使用 parts[-1] 防止偶尔的多余空格报错
                        
            if words:
                # 【修改点 2】最后一句也要做同样的过滤检查
                if any(tag != 0 for tag in tags):
                    self.texts.append(words)
                    self.labels.append(tags)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        words = self.texts[idx]
        tags = self.labels[idx]

        encoding = self.tokenizer(
            words, is_split_into_words=True, return_offsets_mapping=True,
            padding='max_length', truncation=True, max_length=self.max_len
        )

        labels = []
        for i, offset in enumerate(encoding['offset_mapping']):
            if offset[0] == 0 and offset[1] == 0: 
                labels.append(-100) 
            else:
                labels.append(tags[i - 1] if i - 1 < len(tags) else -100)

        item = {key: torch.tensor(val) for key, val in encoding.items() if key != 'offset_mapping'}
        item['labels'] = torch.tensor(labels)
        return item

# ================= 3. 开始训练 =================
def main():
    print("⏳ 正在加载预训练 BERT 模型...")
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(LABEL_LIST))

    print("📚 正在解析训练数据并进行浓缩提纯...")
    train_dataset = MedicalNERDataset(DATA_FILE, tokenizer, MAX_LEN)
    print(f"✅ 提纯完毕！剔除废话后，成功加载 {len(train_dataset)} 条高浓度精华数据！")

    training_args = TrainingArguments(
        output_dir="D:\\code\\C++\\Python\\智能问诊与医疗资源调度系统\\medical_ner_model", 
        num_train_epochs=15,              # 【修改点 3】提纯后数据变少了，轮数加大到 15 轮让它学透
        per_device_train_batch_size=8,    
        logging_steps=10,                 
        save_steps=100,
        learning_rate=2e-5,               
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)

    print("🚀 开始高浓度微调训练！")
    trainer.train()
    
    print("✅ 训练完成！正在保存专属模型...")
    trainer.save_model("D:\\code\\C++\\Python\\智能问诊与医疗资源调度系统\\medical_ner_model_final")
    tokenizer.save_pretrained("D:\\code\\C++\\Python\\智能问诊与医疗资源调度系统\\medical_ner_model_final")
    print("🎉 模型已保存在 D:\\code\\C++\\Python\\智能问诊与医疗资源调度系统\\medical_ner_model_final 文件夹！")

if __name__ == "__main__":
    main()