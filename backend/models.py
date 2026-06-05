from sqlalchemy import Column, Integer, String, Text
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    hashed_password = Column(String(100))
    # 将 128 维的声纹 Embedding 保存为 JSON 字符串或逗号分隔的浮点数
    voiceprint_embedding = Column(Text, nullable=True)
