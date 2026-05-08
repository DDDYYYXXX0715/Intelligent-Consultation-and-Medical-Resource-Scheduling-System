import os
import json
import openai
from dotenv import load_dotenv

# 1. 加载我们刚刚建好的 .env 文件 (也就是打开保险箱)
load_dotenv()

# 2. 安全地拿取 API Key
api_key = os.getenv("DASHSCOPE_API_KEY") 

if not api_key:
    raise ValueError("没有找到 API Key，请检查 .env 文件是否配置正确！")

# 配置你的大模型 API (这里以各大厂通用的 base_url 为例)
client = openai.OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com/v1"
)

class IntelligentConsultation:
    def __init__(self):
        # 系统提示词，定义了 AI 的行为和 JSON 输出格式
        self.system_prompt = """
        你是一个专业的AI辅助分诊医生。你的任务是与患者进行多轮对话，收集症状，进行初步诊断，并评估患者的病情严重程度。

【工作流】
1. 提取患者输入中的症状实体（NER）。
2. 根据已有症状，判断是否可以得出初步诊断。
3. 如果信息不足，生成一句追问（例如询问体温、持续时间等）。
4. 如果信息充足，给出初步诊断结论。
5. 必须对你的诊断给出一个置信度分数（0.0 - 1.0）。

【严重等级划分标准】
1级：轻微不适（如普通感冒初期）
2级：普通疾病（如轻度肠胃炎）
3级：需要按时就医（如发烧39度、持续疼痛）
4级：急症，优先排队（如剧烈腹痛、骨折疑似）
5级：危重，立刻抢救（如胸痛伴呼吸困难、大出血）

【强制输出格式】
你必须且只能输出合法的 JSON 格式，不要包含任何其他说明文字：
{
  "extracted_symptoms": ["症状1", "症状2"], 
  "current_diagnosis": "初步判断的疾病（如果信息不足，填 null）",
  "confidence_score": 0.85, 
  "severity_level": 3, 
  "ai_reply": "你要对患者说的话（追问或给出结论）",
  "is_finished": false 
}
        """
        # 对话历史阈值，防止死循环问诊
        self.max_turns = 6 

    def get_llm_response(self, conversation_history):
        """调用大模型获取结构化响应"""
        try:
            response = client.chat.completions.create(
                model="deepseek-chat", 
                messages=conversation_history,
                temperature=0.2, # 降低温度，保证医疗场景的严谨性和 JSON 输出稳定性
                response_format={ "type": "json_object" } # 强制 JSON 输出 (需模型支持)
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"模型调用或解析失败: {e}")
            # 降级处理，触发高不确定性
            return {"confidence_score": 0.0, "ai_reply": "系统处理异常"}

    def process_patient_input(self, patient_input, session_data):
        """处理单次患者输入"""
        
        # 1. 初始化或加载对话历史
        if "history" not in session_data:
            session_data["history"] = [{"role": "system", "content": self.system_prompt}]
            session_data["turn_count"] = 0
            
        # 把患者的新话语加入历史
        session_data["history"].append({"role": "user", "content": patient_input})
        session_data["turn_count"] += 1

        # 2. 调用大模型
        llm_output = self.get_llm_response(session_data["history"])
        
        # 提取模型给出的各个维度数据
        ai_reply = llm_output.get("ai_reply", "抱歉，我没有理解，请重新描述。")
        confidence = llm_output.get("confidence_score", 0.0)
        symptoms = llm_output.get("extracted_symptoms", [])
        severity = llm_output.get("severity_level", 1)
        is_finished = llm_output.get("is_finished", False)

        # 把 AI 的回复也加入历史，用于下一轮
        session_data["history"].append({"role": "assistant", "content": json.dumps(llm_output, ensure_ascii=False)})

        # 3. 核心创新：不确定性评估与异常拦截逻辑
        is_uncertain = False
        triage_data = None
        status_msg = "chatting"

        # 规则 A：模型自身的置信度过低
        if confidence < 0.65:
            is_uncertain = True
            ai_reply = "您的症状比较复杂，为了安全起见，系统正在为您转接人工专业导诊..."
            status_msg = "transfer_to_human"
            
        # 规则 B：对话轮数过多，依然没有得出结论（防止 AI 变成“十万个为什么”）
        elif session_data["turn_count"] >= self.max_turns and not is_finished:
            is_uncertain = True
            ai_reply = "由于病情特征不明显，系统已为您优先转接人工导诊以进一步确诊。"
            status_msg = "transfer_to_human"

        # 4. 问诊正常结束，生成调度数据
        elif is_finished and not is_uncertain:
            status_msg = "completed"
            triage_data = {
                "symptoms": symptoms,
                "disease": llm_output.get("current_diagnosis"),
                "severity_level": severity,
                "confidence": confidence
            }

        return {
            "status": status_msg,
            "ai_reply": ai_reply,
            "current_symptoms": symptoms,
            "triage_data": triage_data
        }

# ================= 模拟测试 =================
if __name__ == "__main__":
    engine = IntelligentConsultation()
    
    # 模拟 Redis 中的会话状态
    current_session = {} 
    
    # 第一轮
    print("患者: 我从昨天开始肚子一直很痛。")
    res1 = engine.process_patient_input("我从昨天开始肚子一直很痛。", current_session)
    print(f"AI: {res1['ai_reply']} | 提取症状: {res1['current_symptoms']}")
    
    # 第二轮
    print("\n患者: 在右下角，而且按下去更痛，我还吐了一次。")
    res2 = engine.process_patient_input("在右下角，而且按下去更痛，我还吐了一次。", current_session)
    print(f"AI: {res2['ai_reply']} | 状态: {res2['status']}")
    if res2['triage_data']:
        print(f"分诊结果送往调度引擎: 严重等级 {res2['triage_data']['severity_level']}, 疾病预测 {res2['triage_data']['disease']}")