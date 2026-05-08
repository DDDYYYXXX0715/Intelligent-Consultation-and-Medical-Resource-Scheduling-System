import torch
import numpy as np
import time
from hospital_env import HospitalEnv
# 从你的训练脚本中导入 DQN 脑结构
from train_dqn import DQN 

def test_dispatcher():
    print("🏥 智能急诊科调度中枢启动！\n" + "="*45)
    
    # 1. 准备环境和大脑
    env = HospitalEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(state_dim=3, action_dim=3).to(device)
    
    # 2. 加载你刚才训练好的记忆权重
    model.load_state_dict(torch.load("drq_dispatcher_model.pth", map_location=device))
    model.eval() # 开启评估模式（关闭一切随机探索，只用最强实力）
    
    # 翻译 AI 的动作编号
    action_names = [
        "🚨 紧急呼叫 -> 【重症】患者入诊室！", 
        "📢 普通呼叫 -> 【轻症】患者入诊室", 
        "☕ 待命观望 -> (无空闲医生或大厅无人)"
    ]
    
    state, _ = env.reset()
    
    # 模拟医院运行 15 分钟
    for step in range(15):
        print(f"🕒 第 {step+1:02d} 分钟 | 候诊大厅: 重症 {state[0]} 人, 轻症 {state[1]} 人 | 空闲医生: {state[2]} 人")
        
        # 3. AI 观察状态并做出决定
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
            action = q_values.argmax().item() # 直接选最高分的动作
            
        print(f"   🤖 AI 护士长决定: {action_names[action]}")
        
        # 4. 执行动作，进入下一分钟
        next_state, reward, done, _, _ = env.step(action)
        print(f"   (本步调度得分/惩罚: {reward})\n")
        
        state = next_state
        time.sleep(1.5) # 稍微停顿一下，方便你看清它的决策过程
        
        if done:
            print("🎉 大厅已清空，所有患者处理完毕！")
            break

if __name__ == "__main__":
    test_dispatcher()