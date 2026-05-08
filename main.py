from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import BertTokenizerFast, BertForTokenClassification
from sqlalchemy.orm import Session
import datetime
import json
from collections import Counter

# 导入我们的“双擎”数据库配置
from database import SessionLocal, User, TriageQueue, mongo_logs 
from hospital_env import HospitalEnv
from train_dqn import DQN

app = FastAPI(title="智能急诊科问诊与调度中枢", version="4.0 (知识图谱完全体)")
# 1. 记忆中枢：临时记录用户的对话进度（生产环境建议用 Redis）
# 格式: {"wx_user_001": {"step": 1, "history_symptoms": ["腹痛"]}}
user_sessions = {}

# 2. 伴随症状知识图谱 (为了快速跑通Demo，我们手写几个极其常见的急诊关联症状)
# 真实商业项目中，这些数据同样是从你下载的 medical.json 的 "acompany" 字段提炼出来的
# 2. 伴随症状知识图谱 (升级版：加入口语化同义词防线)
accompany_kg = {
    "腹痛": {
        "synonyms": ["肚子疼", "肚子痛", "胃痛", "腹痛"], 
        "related": ["发热", "呕吐", "腹泻", "便血", "恶心"]
    },
    "头痛": {
        "synonyms": ["头疼", "脑袋疼", "头痛"], 
        "related": ["头晕", "恶心", "视力模糊", "发烧", "颈部僵硬"]
    },
    "胸痛": {
        "synonyms": ["胸口疼", "心口疼", "胸痛", "胸闷"], 
        "related": ["呼吸困难", "心悸", "大汗", "左臂放射痛"]
    }
}
# 允许所有的跨域请求（给网页大屏用的）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= 1. AI 模型、环境与【知识图谱】初始化 =================
tokenizer = None
bert_model = None
dqn_model = None
env = None
triage_dict = {}         # 科室分诊指南
medical_symptoms = set() # 精准症状词典

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_LIST = ["O", "B-Symptom", "I-Symptom", "B-Disease", "I-Disease"]
ID_TO_LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

@app.on_event("startup")
def load_models():
    global tokenizer, bert_model, dqn_model, env, triage_dict, medical_symptoms
    print("⏳ 正在加载 AI 医生双核大脑与知识图谱...")
    
    # 1. 加载左脑 BERT
    MODEL_DIR = "./medical_ner_model_final"
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
    bert_model = BertForTokenClassification.from_pretrained(MODEL_DIR)
    
    # 2. 加载右脑 DQN
    dqn_model = DQN(state_dim=3, action_dim=3).to(device)
    dqn_model.load_state_dict(torch.load("drq_dispatcher_model.pth", map_location=device))
    dqn_model.eval()
    
    # 3. 加载知识图谱（你刚炼出来的丹！）
    try:
        with open('./triage_dict.json', 'r', encoding='utf-8') as f:
            triage_dict = json.load(f)
        with open('./dict/dict/symptom.txt', 'r', encoding='utf-8') as f:
            medical_symptoms = set([line.strip() for line in f if line.strip()])
    except Exception as e:
        print(f"⚠️ 警告：知识图谱加载失败，请检查文件是否存在。{e}")

    # 4. 环境初始化
    env = HospitalEnv()
    env.reset()
    env.severe_waiting = 0
    env.mild_waiting = 0
    env.free_doctors = 3
    print("✅ 系统全部启动就绪！开始接诊！")

# ================= 2. 核心症状提取函数 (深度学习 + 字典双保险) =================
def extract_entities(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    entities = {"Symptom": [], "Disease": []}
    current_entity = ""
    current_type = None
    for token, pred_id in zip(tokens, predictions):
        if token in ["[CLS]", "[SEP]", "[PAD]"]: continue
        label = ID_TO_LABEL[pred_id]
        if label.startswith("B-"):
            if current_entity: entities[current_type].append(current_entity)
            current_entity = token.replace("##", "") 
            current_type = label.split("-")[1]
        elif label.startswith("I-") and current_entity:
            current_entity += token.replace("##", "")
        else:
            if current_entity:
                entities[current_type].append(current_entity)
                current_entity = ""
                current_type = None
    if current_entity: entities[current_type].append(current_entity)
    return entities

def get_final_symptoms(text):
    """双剑合璧：BERT语义提取 + 字典精准匹配"""
    # A. 深度学习提取
    bert_results = extract_entities(text)["Symptom"]
    # B. 字典字符串匹配 (兜底防御)
    dict_results = [s for s in medical_symptoms if s in text]
    # 合并并去重
    return list(set(bert_results + dict_results))

# ================= 3. 数据库与前端数据格式 =================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class PatientRequest(BaseModel):
    user_id: str        
    name: str           
    text: str           
    intent: str         

# ================= 4. 核心接口：评估 + 调度 + 双重持久化 =================
@app.post("/api/patient_request")
def handle_patient_request(req: PatientRequest, db: Session = Depends(get_db)):
    global env, user_sessions
    
    # 1. 获取当前用户刚输入的症状
    current_symptoms = get_final_symptoms(req.text)
    
    # 2. 读取用户的“记忆”
    # 如果是第一次说话，默认赋予 step: 1
    session = user_sessions.get(req.user_id, {"step": 1, "history_symptoms": []})

    # ================= [阶段一：追问拦截器] =================
    if session["step"] == 1 and req.intent == "register":
        # 如果什么症状都没提取出来，温柔地让患者重说
        if not current_symptoms:
            return {"status": "success", "type": "question", "message": "对不起，我没有明确识别到您的症状，能请您详细描述一下哪里不舒服吗？（例如：肚子疼、头晕）"}

        # 触发智能追问：如果症状太单一（只有1个），且命中了我们的追问知识库
        # 触发智能追问：支持同义词模糊匹配
        if len(current_symptoms) == 1:
            main_sym = current_symptoms[0]
            matched_related_syms = []
            
            # 遍历知识库，看看患者说的话有没有命中同义词
            for std_sym, data in accompany_kg.items():
                if any(syn in main_sym or main_sym in syn for syn in data["synonyms"]):
                    matched_related_syms = data["related"]
                    break # 命中了就立刻跳出循环
                    
            # 如果成功找到了对应的伴随症状
            if matched_related_syms:
                # 更新用户记忆，进入等待回答状态 (step: 2)
                user_sessions[req.user_id] = {
                    "step": 2,
                    "history_symptoms": current_symptoms 
                }
                question_msg = f"系统检测到您有【{main_sym}】。为了更精准地为您分诊，请问您是否还伴有以下症状：{', '.join(matched_related_syms)}？\n\n(请回复具体的症状，或者直接回复“没有”)"
                
                return {"status": "success", "type": "question", "message": question_msg}

        # 如果没命中任何同义词，放行到最终分诊
        session["history_symptoms"] = current_symptoms
        session["step"] = 2

    # ================= [阶段二：最终联合分诊] =================
    if session["step"] == 2:
        # 将用户历史说过的症状 + 刚才回答的症状 拼在一起！
        all_symptoms = list(set(session["history_symptoms"] + current_symptoms))
        
        # ⚠️ 极其重要：分诊完成，必须清空该用户的记忆，否则他下次挂号还会乱
        if req.user_id in user_sessions:
            del user_sessions[req.user_id]

        # -------- 以下是原汁原味的 AI 决策调度代码 --------
        department = "急诊普通内科/外科"
        # 专家规则防线
        user_text = req.text
        if any(kw in user_text for kw in ["牙", "口腔", "牙髓", "拔牙", "龋齿"]): department = "口腔科急诊"
        elif any(kw in user_text for kw in ["月经", "痛经", "阴道", "怀孕"]): department = "妇产科急诊"
        elif any(kw in user_text for kw in ["眼睛", "视力", "眼红"]): department = "眼科急诊"
        elif any(kw in user_text for kw in ["骨折", "崴脚", "脱臼", "外伤"]): department = "骨科急诊"
        else:
            # 知识图谱投票
            if all_symptoms:
                suggested_depts = [triage_dict[sym] for sym in all_symptoms if sym in triage_dict]
                if suggested_depts: department = Counter(suggested_depts).most_common(1)[0][0]

        # 轻重症判定
        severe_keywords = ["痛", "血", "剧烈", "晕", "烧", "恶心", "吐", "紫绀", "呼吸困难", "骨折"]
        is_severe = any(any(kw in sym for kw in severe_keywords) for sym in all_symptoms)
        triage_level = "重症" if is_severe else "轻症"
        
        if is_severe: env.severe_waiting += 1
        else: env.mild_waiting += 1

        my_queue_ahead = env.severe_waiting + env.mild_waiting - 1

        # DQN 调度
        current_state = env._get_obs()
        state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = dqn_model(state_tensor).argmax().item()
        action_names = ["🚨 呼叫重症患者入诊室", "📢 呼叫轻症患者入诊室", "☕ 医生排满，大厅等待"]
        env.step(action)
        
        final_triage_str = f"【{triage_level}】建议挂号: {department}"
        
        # 数据库写入 (精简展示，务必保留你原来的双写逻辑)
        new_queue = TriageQueue(user_id=req.user_id, triage_level=final_triage_str)
        db.add(new_queue)
        db.commit()
        
        return {
            "status": "success", 
            "type": "register",
            "triage": final_triage_str,
            "queue_ahead": my_queue_ahead, 
            "dispatcher_action": action_names[action]
        }

# ================= 5. 大屏专属接口 =================
@app.get("/api/queue_status")
def get_queue_status(db: Session = Depends(get_db)):
    global env
    waiting_patients = db.query(TriageQueue).filter(TriageQueue.status == "排队中").order_by(TriageQueue.create_time).all()
    queue_list = []
    for p in waiting_patients:
        user = db.query(User).filter(User.user_id == p.user_id).first()
        queue_list.append({
            "id": p.id,
            "name": user.name if user else "未知",
            "triage": p.triage_level,
            "time": p.create_time.strftime("%H:%M:%S")
        })
    return {
        "status": "success",
        "env_status": {"severe": env.severe_waiting, "mild": env.mild_waiting, "doctors": env.free_doctors},
        "queue_list": queue_list
    }
class DirectRegisterRequest(BaseModel):
    user_id: str
    name: str
    department: str

@app.post("/api/direct_register")
def direct_register(req: DirectRegisterRequest, db: Session = Depends(get_db)):
    global env
    
    # 直接挂号的默认按“轻症”处理（重症通常直接进抢救室了）
    triage_level = f"【轻症】建议挂号: {req.department}"
    env.mild_waiting += 1
    
    # 存入数据库
    new_queue = TriageQueue(user_id=req.user_id, triage_level=triage_level)
    db.add(new_queue)
    db.commit()
    
    return {"status": "success", "message": f"成功挂号 {req.department}"}
@app.get("/api/my_status/{user_id}")
def get_my_status(user_id: str, db: Session = Depends(get_db)):
    global env
    
    # 查找该用户最新的排队记录
    record = db.query(TriageQueue).filter(
        TriageQueue.user_id == user_id, 
        TriageQueue.status == "排队中"
    ).order_by(TriageQueue.id.desc()).first()
    
    if not record:
        return {"status": "none"}
        
    # 计算前面的重症人数和轻症人数
    # 为了简化，我们直接返回环境里的当前状态
    severe_count = env.severe_waiting
    
    # 假设：1个重症处理需 15 分钟，1个轻症需 10 分钟
    if "重症" in record.triage_level:
        est_time = severe_count * 15
    else:
        # 如果我是轻症，所有的重症都要排在我前面！
        est_time = severe_count * 15 + env.mild_waiting * 10
        
    return {
        "status": "waiting",
        "triage_level": record.triage_level,
        "severe_ahead": severe_count, # 极其重要：把重症数量传给前端，用于判断是否被插队
        "est_time": est_time,
        "create_time": record.create_time.strftime("%H:%M:%S")
    }
@app.post("/api/doctor_call")
def doctor_call_patient(db: Session = Depends(get_db)):
    global env
    
    # 1. 优先查找排队中的“重症”患者（按挂号时间先后）
    patient_to_call = db.query(TriageQueue).filter(
        TriageQueue.status == "排队中",
        TriageQueue.triage_level.like("%重症%")
    ).order_by(TriageQueue.create_time).first()
    
    # 2. 如果没有重症，再去叫“轻症”患者
    if not patient_to_call:
        patient_to_call = db.query(TriageQueue).filter(
            TriageQueue.status == "排队中",
            TriageQueue.triage_level.like("%轻症%")
        ).order_by(TriageQueue.create_time).first()
        
    # 3. 如果大厅里根本没人
    if not patient_to_call:
        return {"status": "empty", "message": "当前大厅无排队患者"}
        
    # 4. 执行叫号：修改数据库状态
    patient_to_call.status = "已叫号"
    db.commit()
    
    # 5. 更新大屏左侧的统计数字
    if "重症" in patient_to_call.triage_level:
        env.severe_waiting = max(0, env.severe_waiting - 1)
    else:
        env.mild_waiting = max(0, env.mild_waiting - 1)
        
    # 去查一下这个患者的名字，方便大屏播报
    user = db.query(User).filter(User.user_id == patient_to_call.user_id).first()
    patient_name = user.name if user else "未知患者"
        
    return {
        "status": "success", 
        "message": f"请 {patient_name} 到 1 号诊室就诊",
        "called_id": patient_to_call.id
    }