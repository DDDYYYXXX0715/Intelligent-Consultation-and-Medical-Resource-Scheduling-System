import random

class ConsultationEngine:
    def __init__(self):
        # 实际比赛中这里加载微调后的 BERT 或 LLaMA 模型
        # self.model = AutoModelForCausalLM.from_pretrained("your-finetuned-llama")
        pass

    def extract_symptoms_ner(self, text):
        # TODO: 接入命名实体识别模型提取症状
        return ["头痛", "发热"]

    def process_dialogue(self, user_id, message, session_id):
        # 1. 状态追踪 (这里需结合 Redis 提取历史对话)
        symptoms = self.extract_symptoms_ner(message)
        
        # 2. 调用大模型进行疾病预测与置信度评估
        # 伪代码：ai_response, confidence_score = self.model.predict(message)
        ai_response = "初步判断可能为病毒性感染。请问您的体温是多少度？"
        confidence_score = random.uniform(0.5, 0.95) # 模拟置信度
        
        # 3. 不确定性评估 (核心创新点)
        is_uncertain = confidence_score < 0.60 
        
        # 假设收集完症状，生成分诊结果 (1=轻症, 5=重症)
        triage_data = None
        if "39度" in message: # 简单的结束条件示例
            triage_data = {"user_id": user_id, "severity_level": 4, "disease": "高热疑似流感"}
            ai_response = "已为您完成初步诊断，正在为您紧急排号..."

        return ai_response, is_uncertain, triage_data