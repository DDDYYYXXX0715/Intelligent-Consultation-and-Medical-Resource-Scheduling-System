import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from hospital_env import HospitalEnv # 导入你刚才写的医院环境

# ================= 1. 定义大脑 (神经网络) =================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        # 这是一个简单的三层神经网络
        # 输入：3个状态 (重症数, 轻症数, 医生数)
        # 输出：3个动作的打分 (叫重症, 叫轻症, 等待)
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# ================= 2. 核心参数设置 =================
BATCH_SIZE = 64          # 每次从记忆中学习的样本数
GAMMA = 0.99             # 远见程度 (越接近1，AI越看重长远利益)
LR = 0.001               # 学习率
EPSILON_START = 1.0      # 初始探索率 (1.0表示100%瞎蒙试错)
EPSILON_END = 0.01       # 最终探索率 (后期主要靠经验)
EPSILON_DECAY = 0.995    # 探索率衰减速度
MEMORY_CAPACITY = 10000  # 记忆容量 (最多记多少步)
EPISODES = 500           # 让 AI 玩多少局游戏

# ================= 3. 训练主循环 =================
def train():
    env = HospitalEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 正在使用 {device} 训练强化学习调度大脑...")

    # 初始化神经网络和记忆库
    q_network = DQN(state_dim=3, action_dim=3).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    memory = deque(maxlen=MEMORY_CAPACITY)
    
    epsilon = EPSILON_START

    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        
        # 每一局游戏最多玩 50 步，防止陷入死循环
        while not done and step_count < 50:
            step_count += 1
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # --- 动作选择 (探索 vs 利用) ---
            if random.random() < epsilon:
                action = env.action_space.sample() # 瞎蒙探索
            else:
                with torch.no_grad():
                    q_values = q_network(state_tensor)
                    action = q_values.argmax().item() # 靠经验选最高分动作
            
            # --- 与环境交互 ---
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            
            # --- 把经历存入记忆库 ---
            memory.append((state, action, reward, next_state, done))
            state = next_state
            
            # --- 从记忆中抽样学习 (核心炼丹过程) ---
            if len(memory) >= BATCH_SIZE:
                # 随机抽取一批记忆
                batch = random.sample(memory, BATCH_SIZE)
                b_states, b_actions, b_rewards, b_next_states, b_dones = zip(*batch)
                
                b_states = torch.FloatTensor(np.array(b_states)).to(device)
                b_actions = torch.LongTensor(b_actions).unsqueeze(1).to(device)
                b_rewards = torch.FloatTensor(b_rewards).unsqueeze(1).to(device)
                b_next_states = torch.FloatTensor(np.array(b_next_states)).to(device)
                b_dones = torch.FloatTensor(b_dones).unsqueeze(1).to(device)
                
                # 计算当前 Q 值
                current_q = q_network(b_states).gather(1, b_actions)
                
                # 计算目标 Q 值
                with torch.no_grad():
                    max_next_q = q_network(b_next_states).max(1)[0].unsqueeze(1)
                    target_q = b_rewards + GAMMA * max_next_q * (1 - b_dones)
                
                # 反向传播更新神经网络
                loss = loss_fn(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        # --- 每一局结束后的操作 ---
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY) # 慢慢减少瞎蒙的概率
        
        # 每隔 50 局打印一次战况
        if (episode + 1) % 50 == 0:
            print(f"🎮 游戏局数: {episode+1:3d}/{EPISODES} | "
                  f"本局总得分: {total_reward:6.1f} | "
                  f"当前瞎蒙率(Epsilon): {epsilon:.2f}")

    print("\n✅ 训练完成！正在保存调度大脑的权重...")
    torch.save(q_network.state_dict(), "drq_dispatcher_model.pth")
    print("🎉 模型已保存为 drq_dispatcher_model.pth！你的智能调度中心上线了！")

if __name__ == "__main__":
    train()