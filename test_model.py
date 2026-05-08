import torch
from transformers import BertTokenizerFast, BertForTokenClassification

# ================= 1. 加载我们刚炼好的专属大模型 =================
MODEL_DIR = "D:\\code\\C++\\Python\\智能问诊与医疗资源调度系统\\medical_ner_model_final"
print(f"⏳ 正在加载模型权重，请稍候...")
tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
model = BertForTokenClassification.from_pretrained(MODEL_DIR)

# 标签字典 (必须和训练时完全一致)
LABEL_LIST = ["O", "B-Symptom", "I-Symptom", "B-Disease", "I-Disease"]
ID_TO_LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

def extract_entities(text):
    """把一段话喂给模型，吐出症状和疾病"""
    
    # 2. 把文字变成数字 ID 
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    # 3. 模型进行预测 (前向传播)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 获取每个字概率最大的那个标签 ID
    predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
    
    # 获取模型眼中的文字列表 (把 ID 还原回汉字)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # 4. 解码 BIO 标签，拼接成人类看得懂的词语
    entities = {"Symptom": [], "Disease": []}
    current_entity = ""
    current_type = None
    
    for token, pred_id in zip(tokens, predictions):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
            
        label = ID_TO_LABEL[pred_id]
        
        # 遇到 B (Begin)，说明是一个新实体的开始
        if label.startswith("B-"):
            if current_entity: # 如果之前有存货，先放进列表
                entities[current_type].append(current_entity)
            current_entity = token.replace("##", "") 
            current_type = label.split("-")[1]
            
        # 遇到 I (Inside)，说明是实体的中间部分，继续拼接
        elif label.startswith("I-") and current_entity:
            current_entity += token.replace("##", "")
            
        # 遇到 O (Outside)，说明是普通废话，结束拼接
        else:
            if current_entity:
                entities[current_type].append(current_entity)
                current_entity = ""
                current_type = None
                
    # 收尾，防止最后一句话结束时漏掉词
    if current_entity:
        entities[current_type].append(current_entity)
        
    return entities

# ================= 5. 测试环节 =================
if __name__ == "__main__":
    print("\n🏥 智能问诊 NER 引擎测试台启动！\n" + "="*40)
    
    # 这里我准备了几个非常有迷惑性的句子，你可以随便修改它们！
    test_texts = [
        "我昨天晚上突然右下腹剧烈疼痛，还伴有低烧，还吐了一次。",
        "孩子鸡鸡旁边肿了一个包，去医院看说是鞘膜积液或者斜疝，需要做手术吗？",
        "医生你好，我最近总是头晕恶心，是不是得了高血压啊？"
    ]
    
    for text in test_texts:
        print(f"🗣️ 患者自述: {text}")
        result = extract_entities(text)
        print(f"🤖 抓取症状: {result['Symptom']}")
        print(f"🤖 抓取疾病: {result['Disease']}\n" + "-"*40)