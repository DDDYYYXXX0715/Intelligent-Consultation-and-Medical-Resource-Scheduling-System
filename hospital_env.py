import gymnasium as gym
from gymnasium import spaces
import numpy as np

class HospitalEnv(gym.Env):
    """
    智能医疗资源调度环境模拟器
    """
    def __init__(self):
        super(HospitalEnv, self).__init__()
        
        # 定义系统的最大容量（防止数组越界）
        self.max_patients = 10
        self.max_doctors = 3
        
        # 1. 定义动作空间 (Action Space): AI 可以做的选择
        # 0: 叫号重症患者 (Severe)
        # 1: 叫号轻症患者 (Mild)
        # 2: 保持观望 (Do nothing / 没有空闲医生)
        self.action_space = spaces.Discrete(3)
        
        # 2. 定义状态空间 (Observation Space): AI 能看到的信息
        # [重症排队人数, 轻症排队人数, 空闲医生数量]
        self.observation_space = spaces.MultiDiscrete(
            [self.max_patients + 1, self.max_patients + 1, self.max_doctors + 1]
        )
        
        # 初始化状态变量
        self.severe_waiting = 0
        self.mild_waiting = 0
        self.free_doctors = self.max_doctors

    def reset(self, seed=None, options=None):
        """每回合开始时，重置医院状态（类似于游戏重新开局）"""
        super().reset(seed=seed)
        
        # 随机生成初始的排队人数
        self.severe_waiting = np.random.randint(1, 5)
        self.mild_waiting = np.random.randint(2, 8)
        self.free_doctors = self.max_doctors
        
        return self._get_obs(), {}

    def _get_obs(self):
        """获取当前状态"""
        return np.array([self.severe_waiting, self.mild_waiting, self.free_doctors], dtype=np.int32)

    def step(self, action):
        """
        AI 做出动作后，环境的推进逻辑
        返回: (下一个状态, 奖励, 是否结束, 是否截断, 额外信息)
        """
        reward = 0
        done = False
        info = {}
        
        # --- 处理 AI 的动作 ---
        if action == 0: # 尝试叫号重症
            if self.severe_waiting > 0 and self.free_doctors > 0:
                self.severe_waiting -= 1
                self.free_doctors -= 1
                reward += 5 # 成功调度重症，给予正反馈
            else:
                reward -= 2 # 乱叫号（没人或者没医生），给予惩罚
                
        elif action == 1: # 尝试叫号轻症
            if self.mild_waiting > 0 and self.free_doctors > 0:
                self.mild_waiting -= 1
                self.free_doctors -= 1
                # 如果放着重症不管去叫轻症，要严厉惩罚！
                if self.severe_waiting > 0:
                    reward -= 10 
                else:
                    reward += 1
            else:
                reward -= 2
        elif action == 2: # AI 决定待命观望
            # 如果医生有空，且外面有病人（无论轻重），AI 却选择观望不作为
            if self.free_doctors > 0 and (self.severe_waiting > 0 or self.mild_waiting > 0):
                reward -= 15 # 给予超级严厉的“渎职惩罚”！比叫错号罚得还狠！        
        # --- 模拟时间推移与状态变化 ---
        # 模拟医生看完病空闲下来 (简化逻辑：每步有一定概率释放医生)
        if self.free_doctors < self.max_doctors and np.random.rand() < 0.3:
            self.free_doctors += 1
            
        # 模拟新的患者不断挂号进来
        if self.severe_waiting < self.max_patients and np.random.rand() < 0.1:
            self.severe_waiting += 1
        if self.mild_waiting < self.max_patients and np.random.rand() < 0.2:
            self.mild_waiting += 1

        # --- 计算核心奖励（惩罚所有等待的人） ---
        # 重症等一分钟的代价是轻症的 5 倍
        reward -= (self.severe_waiting * 5 + self.mild_waiting * 1)
        
        # 如果候诊大厅全空了，这局游戏完美通关
        if self.severe_waiting == 0 and self.mild_waiting == 0:
            reward += 50
            done = True
            
        return self._get_obs(), reward, done, False, info

# ================= 测试环境 =================
if __name__ == "__main__":
    env = HospitalEnv()
    obs, _ = env.reset()
    print(f"🏥 初始状态 [重症数, 轻症数, 空闲医生]: {obs}")
    
    # 我们先让一个“随机脑”的 AI 随便玩 5 步，看看会发生什么
    for i in range(5):
        # AI 随机瞎选一个动作 (0, 1, 2)
        random_action = env.action_space.sample() 
        obs, reward, done, _, _ = env.step(random_action)
        
        print(f"步骤 {i+1} | AI 动作: {random_action} | 当前状态: {obs} | 本步得分: {reward}")
        
        if done:
            print("🎉 大厅清空，游戏结束！")
            break