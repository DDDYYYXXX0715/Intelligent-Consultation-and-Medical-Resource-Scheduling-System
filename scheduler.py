import time
import heapq

class SchedulingEngine:
    def __init__(self):
        # 优先级队列，存放 (优先级, 时间戳, user_id)
        # Python 的 heapq 是小顶堆，所以优先级可以用负数表示重症（-5 > -1）
        self.waiting_queue = []
        self.doctor_loads = {"doc_1": 2, "doc_2": 5, "doc_3": 1} # 医生当前负载

    def allocate_resource(self, triage_data):
        user_id = triage_data["user_id"]
        severity = triage_data["severity_level"]
        
        # DRL 进阶方法预留接口
        # agent_action = self.drl_agent.predict(state=[severity, doctor_loads, current_queue])
        
        # 传统/基础方法：重症优先队列 + 简单负载均衡
        priority = -severity 
        entry_time = time.time()
        
        # 将患者加入优先级队列
        heapq.heappush(self.waiting_queue, (priority, entry_time, user_id))
        
        # 寻找最空闲的医生 (负载均衡)
        best_doctor = min(self.doctor_loads, key=self.doctor_loads.get)
        self.doctor_loads[best_doctor] += 1
        
        # 计算预估等待时间 (根据队列前方人数和严重程度动态计算)
        estimated_wait_minutes = self.calculate_wait_time(priority)

        return {
            "assigned_doctor": best_doctor,
            "queue_number": len(self.waiting_queue),
            "estimated_wait_time": f"{estimated_wait_minutes} 分钟",
            "priority_level": severity
        }

    def calculate_wait_time(self, priority):
        # 越严重的病人，插队越靠前，等待时间越短
        base_time = len(self.waiting_queue) * 10
        discount = abs(priority) * 5 
        return max(0, base_time - discount)