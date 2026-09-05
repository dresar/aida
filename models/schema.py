from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base, engine

class Customer(Base):
    __tablename__ = "customers"

    customer_id = Column(String, primary_key=True, index=True)
    gender = Column(String)
    senior_citizen = Column(Integer)
    partner = Column(String)
    dependents = Column(String)
    tenure = Column(Integer)
    phone_service = Column(String)
    multiple_lines = Column(String)
    internet_service = Column(String)
    online_security = Column(String)
    online_backup = Column(String)
    device_protection = Column(String)
    tech_support = Column(String)
    streaming_tv = Column(String)
    streaming_movies = Column(String)
    contract = Column(String)
    paperless_billing = Column(String)
    payment_method = Column(String)
    monthly_charges = Column(Float)
    total_charges = Column(Float)
    churn = Column(String) # 'Yes' or 'No' (Ground Truth)

    predictions = relationship("Prediction", back_populates="customer")

class Prediction(Base):
    __tablename__ = "predictions"

    prediction_id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String, ForeignKey("customers.customer_id"))
    churn_prediction = Column(String) # 'Yes' or 'No'
    churn_probability = Column(Float)
    risk_category = Column(String) # Low, Medium, High
    created_at = Column(DateTime, default=datetime.utcnow)

    customer = relationship("Customer", back_populates="predictions")

class ModelMetrics(Base):
    __tablename__ = "model_metrics"

    run_id = Column(Integer, primary_key=True, index=True)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    roc_auc = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    confusion_matrix = relationship("ConfusionMatrix", back_populates="metric", uselist=False)
    feature_importances = relationship("FeatureImportance", back_populates="metric")

class ConfusionMatrix(Base):
    __tablename__ = "confusion_matrix"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("model_metrics.run_id"))
    true_negative = Column(Integer)
    false_positive = Column(Integer)
    false_negative = Column(Integer)
    true_positive = Column(Integer)

    metric = relationship("ModelMetrics", back_populates="confusion_matrix")

class FeatureImportance(Base):
    __tablename__ = "feature_importance"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("model_metrics.run_id"))
    feature_name = Column(String)
    importance_score = Column(Float)

    metric = relationship("ModelMetrics", back_populates="feature_importances")

class RetentionRecommendation(Base):
    __tablename__ = "retention_recommendations"

    id = Column(Integer, primary_key=True, index=True)
    risk_category = Column(String, unique=True)
    recommendation_text = Column(Text)

class ModelRegistry(Base):
    __tablename__ = "model_registry"
    
    id = Column(Integer, primary_key=True, index=True)
    version = Column(String, unique=True)
    algorithm = Column(String)
    hyperparameters = Column(Text) # JSON string
    filepath = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    accuracy = Column(Float)
    f1_score = Column(Float)
    is_active = Column(Integer, default=0)

class TrainingLog(Base):
    __tablename__ = "training_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    level = Column(String) # INFO, WARNING, ERROR
    message = Column(Text)

# Create tables
if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")
