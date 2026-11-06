import os
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get("postgresql://neondb_owner:npg_Lr0qZben3Aym@ep-purple-union-beddc5qe-pooler.c-9.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require")

# Si no hay variable de entorno (desarrollo local), usa SQLite como fallback temporal
if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./local_fleet.db"

# Corrección de protocolo para psycopg v3
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

# Asegurar reconexión automática tras suspensión con pool_pre_ping=True
if "sqlite" in DATABASE_URL:
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class TelemetryRecord(Base):
    __tablename__ = "telemetry"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    robot_id = Column(String, index=True, nullable=False)
    timestamp = Column(Float, nullable=False)
    x = Column(Float, nullable=True)
    y = Column(Float, nullable=True)
    speed = Column(Float, nullable=False)
    battery = Column(Float, nullable=False)
    motor_temp = Column(Float, nullable=False)
    vibration = Column(Float, nullable=False)
    motor_current = Column(Float, nullable=False)
    obstacle_distance = Column(Float, nullable=False)
    status = Column(String, nullable=True)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()