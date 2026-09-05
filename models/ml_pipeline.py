import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sqlalchemy.orm import Session
import joblib
import os
import shutil
import json
import uuid
from datetime import datetime
from .schema import Customer, ModelMetrics, ConfusionMatrix, FeatureImportance, ModelRegistry, TrainingLog

class ChurnPipeline:
    def __init__(self, db_session: Session = None):
        self.db = db_session
        self.model = None
        
        # Define features
        self.numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        self.categorical_features = [
            'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod', 'SeniorCitizen'
        ]

    def log(self, message, level="INFO"):
        if self.db:
            log_entry = TrainingLog(level=level, message=message)
            self.db.add(log_entry)
            self.db.commit()
        print(f"[{level}] {message}")

    def load_data_from_db(self):
        if not self.db:
            raise ValueError("Database session not initialized")
        
        customers = self.db.query(Customer).all()
        if not customers:
            return pd.DataFrame()
            
        data = [{
            'customerID': c.customer_id,
            'gender': c.gender,
            'SeniorCitizen': c.senior_citizen,
            'Partner': c.partner,
            'Dependents': c.dependents,
            'tenure': c.tenure,
            'PhoneService': c.phone_service,
            'MultipleLines': c.multiple_lines,
            'InternetService': c.internet_service,
            'OnlineSecurity': c.online_security,
            'OnlineBackup': c.online_backup,
            'DeviceProtection': c.device_protection,
            'TechSupport': c.tech_support,
            'StreamingTV': c.streaming_tv,
            'StreamingMovies': c.streaming_movies,
            'Contract': c.contract,
            'PaperlessBilling': c.paperless_billing,
            'PaymentMethod': c.payment_method,
            'MonthlyCharges': c.monthly_charges,
            'TotalCharges': c.total_charges,
            'Churn': c.churn
        } for c in customers]
        
        return pd.DataFrame(data)

    def preprocess(self, df):
        # Handle TotalCharges missing/errors
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
        
        X = df.drop(['Churn', 'customerID'], axis=1, errors='ignore')
        y = None
        if 'Churn' in df.columns:
            y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
            
        return X, y

    def build_pipeline(self, tuning=False):
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            verbose_feature_names_out=False
        )
        
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', XGBClassifier(
                eval_metric='logloss',
                random_state=42
            ))
        ])
        
        return pipeline

    def load_data_from_datasets_folder(self):
        datasets_dir = "datasets"
        if not os.path.exists(datasets_dir):
            self.log(f"Direktori {datasets_dir} tidak ditemukan.", "WARNING")
            return pd.DataFrame()
            
        all_files = [os.path.join(datasets_dir, f) for f in os.listdir(datasets_dir) if f.endswith('.csv') or f.endswith('.xlsx')]
        
        if not all_files:
            self.log("Tidak ada file dataset (CSV/Excel) di folder datasets.", "WARNING")
            return pd.DataFrame()
            
        self.log(f"Menemukan {len(all_files)} file dataset. Memulai penggabungan...")
        
        dfs = []
        for file in all_files:
            try:
                if file.endswith('.csv'):
                    # Try utf-8 then latin-1
                    try:
                        df = pd.read_csv(file, encoding='utf-8')
                    except UnicodeDecodeError:
                        df = pd.read_csv(file, encoding='latin-1')
                else:
                    df = pd.read_excel(file)
                
                # Normalize columns if needed (simple mapping)
                # Map PascalCase to match what preprocess expects (if needed, but preprocess uses df columns directly)
                # Actually, our preprocess expects 'Churn' and drops 'customerID'. 
                # Let's standardize column names to match what the pipeline expects.
                # Pipeline expects: tenure, MonthlyCharges, TotalCharges, etc.
                # The uploaded files usually have PascalCase.
                
                dfs.append(df)
                self.log(f"Berhasil memuat: {os.path.basename(file)} ({len(df)} baris)")
            except Exception as e:
                self.log(f"Gagal memuat {os.path.basename(file)}: {str(e)}", "ERROR")
                
        if not dfs:
            return pd.DataFrame()
            
        full_df = pd.concat(dfs, ignore_index=True)
        self.log(f"Total data gabungan: {len(full_df)} baris")
        return full_df

    def train_from_datasets_folder(self, n_iter=5):
        self.log("Memulai proses training dari folder datasets...")
        
        df = self.load_data_from_datasets_folder()
        if df.empty:
            self.log("Dataset kosong. Aborting training.", "ERROR")
            return None
            
        # Ensure directory exists for model artifacts
        if not os.path.exists('model_storage'):
            os.makedirs('model_storage')
            
        X, y = self.preprocess(df)
        
        if y is None:
             self.log("Kolom Target 'Churn' tidak ditemukan di dataset.", "ERROR")
             return None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        pipeline = self.build_pipeline()
        
        # Hyperparameter Space
        param_dist = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__subsample': [0.8, 1.0]
        }
        
        self.log("Menjalankan RandomizedSearchCV (3-Fold CV)...")
        search = RandomizedSearchCV(
            pipeline, 
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring='f1',
            cv=3, 
            verbose=1,
            random_state=42,
            n_jobs=1  # Changed from -1 to 1 to avoid serialization issues
        )
        
        search.fit(X_train, y_train)
        
        best_model = search.best_estimator_
        best_params = search.best_params_
        self.log(f"Parameter Terbaik: {json.dumps(best_params)}")
        
        # Evaluate
        self.log("Evaluasi model pada Test Set...")
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
        
        self.log(f"Hasil Evaluasi: Akurasi={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
        
        # Save Model Artifact (SINGLE MASTER MODEL)
        # Use a timestamp for versioning in registry, but overwrite the file for "Brain Growth"
        version_id = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        master_model_path = "model_storage/churn_model_master.pkl"
        
        # Backup previous master if exists (optional, good practice)
        if os.path.exists(master_model_path):
            shutil.copy(master_model_path, f"model_storage/churn_model_backup_{version_id}.pkl")
            
        joblib.dump(best_model, master_model_path)
        self.log(f"Model Master diperbarui: {master_model_path}")
        
        # Save to Registry
        registry_entry = ModelRegistry(
            version=version_id,
            algorithm="XGBoost + SMOTE (Incremental Growth)",
            hyperparameters=json.dumps(best_params),
            filepath=master_model_path,
            accuracy=metrics['accuracy'],
            f1_score=metrics['f1_score'],
            is_active=1
        )
        self.db.add(registry_entry)
        
        # Save Metrics & CM (Legacy Support)
        self._save_legacy_metrics(metrics, y_test, y_pred, best_model)
        
        # Update active model symlink/file (Keep for compatibility)
        joblib.dump(best_model, 'churn_model.pkl')
        
        self.db.commit()
        self.log("Training Selesai. Otak model semakin pintar!")
        return metrics

    def train_with_tuning(self, n_iter=5):
        self.log("Memulai proses training dengan Hyperparameter Tuning...")
        
        df = self.load_data_from_db()
        if df.empty:
            self.log("Tidak ada data di database.", "ERROR")
            return None
        
        self.log(f"Data dimuat: {len(df)} baris.")
        X, y = self.preprocess(df)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        pipeline = self.build_pipeline()
        
        # Hyperparameter Space
        param_dist = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__subsample': [0.8, 1.0]
        }
        
        self.log("Menjalankan RandomizedSearchCV (5-Fold CV)...")
        search = RandomizedSearchCV(
            pipeline, 
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring='f1',
            cv=3, # Reduced CV for speed
            verbose=1,
            random_state=42,
            n_jobs=1  # Changed from -1 to 1 to avoid serialization issues
        )
        
        search.fit(X_train, y_train)
        
        best_model = search.best_estimator_
        best_params = search.best_params_
        self.log(f"Parameter Terbaik: {json.dumps(best_params)}")
        
        # Evaluate
        self.log("Evaluasi model pada Test Set...")
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
        
        self.log(f"Hasil Evaluasi: Akurasi={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
        
        # Save Model Artifact
        version_id = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_filename = f"model_storage/{version_id}.pkl"
        joblib.dump(best_model, model_filename)
        self.log(f"Model disimpan ke: {model_filename}")
        
        # Save to Registry
        registry_entry = ModelRegistry(
            version=version_id,
            algorithm="XGBoost + SMOTE",
            hyperparameters=json.dumps(best_params),
            filepath=model_filename,
            accuracy=metrics['accuracy'],
            f1_score=metrics['f1_score'],
            is_active=1
        )
        self.db.add(registry_entry)
        
        # Save Metrics & CM (Legacy Support)
        self._save_legacy_metrics(metrics, y_test, y_pred, best_model)
        
        # Update active model symlink/file
        joblib.dump(best_model, 'churn_model.pkl')
        
        self.db.commit()
        self.log("Training Selesai.")
        return metrics

    def _save_legacy_metrics(self, metrics, y_test, y_pred, model):
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        db_metrics = ModelMetrics(
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_score=metrics['f1_score'],
            roc_auc=metrics['roc_auc']
        )
        self.db.add(db_metrics)
        self.db.flush() 
        
        db_cm = ConfusionMatrix(
            run_id=db_metrics.run_id,
            true_negative=int(tn),
            false_positive=int(fp),
            false_negative=int(fn),
            true_positive=int(tp)
        )
        self.db.add(db_cm)
        
        # Feature Importance
        try:
            xg_model = model.named_steps['classifier']
            preprocessor = model.named_steps['preprocessor']
            feature_names = preprocessor.get_feature_names_out()
            importances = xg_model.feature_importances_
            
            for name, imp in zip(feature_names, importances):
                if imp > 0.001: 
                    self.db.add(FeatureImportance(
                        run_id=db_metrics.run_id,
                        feature_name=name,
                        importance_score=float(imp)
                    ))
        except Exception as e:
            self.log(f"Gagal simpan feature importance: {e}", "WARNING")

    def predict_one(self, data_dict):
        if os.path.exists('churn_model.pkl'):
            pipeline = joblib.load('churn_model.pkl')
        else:
            raise Exception("Model belum dilatih.")
        
        df = pd.DataFrame([data_dict])
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
        df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
        
        prob = pipeline.predict_proba(df)[0, 1]
        pred = 1 if prob > 0.5 else 0
        
        risk = "Low"
        if prob > 0.7: risk = "High"
        elif prob > 0.3: risk = "Medium"
            
        return {
            'churn_prediction': 'Yes' if pred == 1 else 'No',
            'churn_probability': float(prob),
            'risk_category': risk
        }
