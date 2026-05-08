# 🏥 智能问诊与医疗资源调度系统 (DRQ-LLM)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-teal)
![Vue3](https://img.shields.io/badge/Vue-3.x-brightgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)

本项目围绕“健康数据洞察”主题，打造了一个基于**大语言模型 (LLM)、知识图谱 (KG) 与深度强化学习 (DQN)** 的“双脑协同”急诊调度中枢。

系统旨在解决传统急诊室“分诊依赖人工经验、排队队列僵化、患者信息黑洞”等痛点，实现从患者多轮智能问诊、精准分诊，到急诊大厅动态优先级叫号的全栈业务闭环。

---

## 🏗️ 核心架构

本项目采用“左脑定性，右脑定量”的混合架构：
- **🧠 左脑 (智能分诊)**：采用 `专家规则 + BERT 实体抽取 + 千万级医疗知识图谱`。既保证了专科分诊的绝对安全，又具备强大的口语化语义泛化能力。
- **🧠 右脑 (动态调度)**：采用 `DQN (Deep Q-Network)` 强化学习算法。实时感知急诊大厅负载，动态计算“重症优先与整体周转率”的最优解，打破僵化的先来后到规则。
- **💻 交互端**：患者端（微信小程序，支持语音/多轮追问），医院端（Vue3 + ECharts 实时数据可视化监控大屏）。

---

## 📂 目录结构与文件说明

本项目代码按功能模块划分为以下几部分：

### 1. 🧠 AI 模型训练与数据工程 (AI & Data Engineering)
*本模块包含从大模型数据蒸馏、知识图谱提取到强化学习训练的全套脚本。*

- `medical.json`：原始千万级中文医疗知识图谱数据（脏数据）。
- `build_dict.py`：数据清洗与 ETL 脚本，将大体积 JSON 压缩提炼为轻量级的分诊映射字典。
- `triage_dict.json`：由上述脚本生成的轻量级急诊科室映射字典，供分诊引擎极速调用。
- `auto_labeling.py`：**数据标注引擎**。调用 DeepSeek 大模型 API，自动将患者口语化文本标注为 NER 训练所需的 BIO 格式（知识蒸馏）。
- `train_bio.txt`：由大模型自动生成的高质量 BERT NER 训练集。
- `train_ner.py`：**左脑训练**。使用 `train_bio.txt` 微调 BERT 命名实体识别模型，赋予系统“听懂白话”的能力。
- `hospital_env.py`：**强化学习环境**。模拟急诊大厅（重症排队、轻症排队、空闲医生数）的物理运转规律与 Reward 惩罚机制。
- `train_dqn.py`：**右脑训练**。基于上述环境训练 DQN 调度模型。
- `drq_dispatcher_model.pth`：训练完成的 DQN 强化学习模型权重文件。
- `test_model.py` / `test_dqn.py`：用于在独立环境中测试验证 BERT 模型和 DQN 模型准确率的脚本。

### 2. ⚙️ 中枢后端服务 (FastAPI Backend)
*本模块为系统的神经中枢，负责处理前后端交互、模型推理与状态机管理。*

- `main.py`：**主程序入口**。基于 FastAPI 构建，串联了 BERT 推理、知识图谱匹配、多轮对话追问（Session状态机）以及 DQN 调度接口。
- `database.py`：数据库连接配置。封装了 MySQL（处理 TriageQueue 挂号排队流转）与 MongoDB（存储系统操作日志）的双写逻辑。
- `ServiceConsultation.py`：问诊业务逻辑层。封装了核心的分诊规则防线与服务层代码。
- `scheduler.py`：辅助调度模块。包含部分定时任务或传统的队列降级调度逻辑。
- `System Prompt.py`：大模型 Prompt 提示词工程文件，用于统一定义 LLM 的人设与输出边界。
- `check_data.py`：后端数据格式探针与完整性校验工具。

### 3. 🖥️ 医院监控大屏 (Frontend Dashboard)
- `dashboard.html`：急诊总控中心指挥大屏。采用 `Vue3 + ECharts` 纯前端实现。
  - 具备实时轮询能力，动态展示全局负载。
  - 内置防崩溃解析防线，支持医生“模拟叫号”与全屏闪烁播报动效。

### 4. 🔧 配置文件 (Configs)
- `.env`：环境变量配置文件（存放数据库 URI、大模型 API Key 等敏感信息，已在 `.gitignore` 中忽略）。
- `.gitignore`：Git 忽略文件配置，防止大型模型权重或敏感配置泄露。

---

## 🚀 快速启动 (Quick Start)

### 1. 环境安装
```bash
git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
cd your-repo-name
pip install -r requirements.txt
```
### 2. 数据库配置
请在根目录创建 `.env` 文件，并配置您的 MySQL 和 MongoDB 连接信息：
```ini
MYSQL_URL=mysql+pymysql://user:password@localhost/hospital
MONGO_URI=mongodb://localhost:27017/
```
### 启动后端中枢
```Bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
服务启动后，可访问 http://127.0.0.1:8000/docs 查看完整的 Swagger API 接口文档。
### 4. 启动前端大屏
直接使用现代浏览器（推荐 Chrome/Edge）双击打开根目录下的 dashboard.html，即可体验带有数据实时互动的可视化大屏。
## 🏆 项目亮点 (Highlights)
  - 彻底拒绝 AI 幻觉：采用 专家规则 -> BERT 抽取 -> 图谱对齐 的三级防线，兼顾泛化性与极高的医疗严谨性。

  - 大模型知识蒸馏：利用 DeepSeek 构造高质量语料并训练私有小模型，大幅降低了线上推理的延迟与成本。

  - 有温度的动态排队：DQN 算法动态优先重症的同时，系统具备重症插队安抚弹窗机制，用数据透明化解医患焦虑。
