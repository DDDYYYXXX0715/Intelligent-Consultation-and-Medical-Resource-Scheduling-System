from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import pymongo
import datetime

# ==========================================
# 🧱 核心引警 1：MySQL (关系型，负责严谨业务)
# ==========================================
# 请把 root:123456 换成你自己的 MySQL 账号密码
# 并且确保 MySQL 里已经有一个叫 hospital_db 的数据库
MYSQL_URL = "mysql+pymysql://root:dongjiayin668@127.0.0.1:3306/hospital_db"

engine = create_engine(MYSQL_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 👨‍⚕️ MySQL 表 1：用户表 (结构极其固定)
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(50), unique=True, index=True) 
    name = Column(String(50))
    register_time = Column(DateTime, default=datetime.datetime.utcnow)

# 🎫 MySQL 表 2：排队业务表 (只存挂号状态，不存长文本)
class TriageQueue(Base):
    __tablename__ = "triage_queue"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(50), index=True)
    triage_level = Column(String(20))          # 重症 / 轻症
    status = Column(String(20), default="排队中") # 排队中 / 已就诊
    create_time = Column(DateTime, default=datetime.datetime.utcnow)

# 自动在 MySQL 中建表
Base.metadata.create_all(bind=engine)


# ==========================================
# 🧬 核心引警 2：MongoDB (文档型，负责海量数据)
# ==========================================
# 默认连接本地 27017 端口
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
# 自动创建名为 hospital_ai 的数据库
mongo_db = mongo_client["hospital_ai"]
# 自动创建名为 consultation_logs 的集合 (用来当问诊记录大本营)
mongo_logs = mongo_db["consultation_logs"]