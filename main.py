from fastapi import FastAPI, Request, Form, Depends, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models
import pandas as pd
import io
import os
import shutil
from models.ml_pipeline import ChurnPipeline
from models.retention_logic import get_retention_strategy
from math import ceil
import httpx
import requests
import json
import asyncio
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Create tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Telco Churn AI")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- API Endpoints for AJAX ---

@app.get("/api/customers")
async def get_customers(page: int = 1, limit: int = 20, db: Session = Depends(get_db)):
    total = db.query(models.Customer).count()
    offset = (page - 1) * limit
    customers = db.query(models.Customer).offset(offset).limit(limit).all()
    
    data = [{
        'customer_id': c.customer_id,
        'gender': c.gender,
        'contract': c.contract,
        'tenure': c.tenure,
        'monthly_charges': c.monthly_charges,
        'total_charges': c.total_charges,
        'churn': c.churn
    } for c in customers]
    
    return {
        "data": data,
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": ceil(total / limit)
    }

@app.get("/api/training/logs")
async def get_training_logs(limit: int = 50, db: Session = Depends(get_db)):
    logs = db.query(models.TrainingLog).order_by(models.TrainingLog.timestamp.desc()).limit(limit).all()
    return [{"timestamp": l.timestamp.isoformat(), "level": l.level, "message": l.message} for l in logs]

@app.delete("/api/training/logs")
async def clear_training_logs(db: Session = Depends(get_db)):
    try:
        db.query(models.TrainingLog).delete()
        db.commit()
        return {"message": "Logs cleared"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def get_models(db: Session = Depends(get_db)):
    models_list = db.query(models.ModelRegistry).order_by(models.ModelRegistry.created_at.desc()).all()
    return [{
        "version": m.version,
        "algorithm": m.algorithm,
        "accuracy": m.accuracy,
        "f1_score": m.f1_score,
        "created_at": m.created_at.isoformat(),
        "is_active": m.is_active
    } for m in models_list]

@app.get("/api/models/download/{version}")
async def download_model(version: str, db: Session = Depends(get_db)):
    model_entry = db.query(models.ModelRegistry).filter_by(version=version).first()
    if not model_entry or not os.path.exists(model_entry.filepath):
        raise HTTPException(status_code=404, detail="Model not found")
    return FileResponse(model_entry.filepath, filename=f"churn_model_{version}.pkl")

@app.get("/health/db")
async def health_check_db(db: Session = Depends(get_db)):
    try:
        # Simple query to check connection
        # Use text() to execute raw SQL safely
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        return {"status": "ok", "message": "Database connected"}
    except Exception as e:
        # Log the actual error for debugging
        print(f"Health Check Failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Database connection failed: {str(e)}")

# --- Background Task ---
def run_training_task():
    db = SessionLocal()
    try:
        pipeline = ChurnPipeline(db)
        # Train using datasets folder
        pipeline.train_from_datasets_folder(n_iter=5)
    except Exception as e:
        if 'pipeline' in locals():
            pipeline.log(f"Training Crash: {str(e)}", "ERROR")
        else:
            print(f"Training Crash (No Pipeline): {str(e)}")
    finally:
        db.close()

@app.post("/api/train")
async def trigger_training(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_training_task)
    return {"message": "Training started in background. Check logs for progress."}

# --- Pages ---

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/analysis", response_class=HTMLResponse)
async def analysis_page(request: Request, db: Session = Depends(get_db)):
    # Get latest metrics
    latest_metrics = db.query(models.ModelMetrics).order_by(models.ModelMetrics.created_at.desc()).first()
    
    cm = None
    if latest_metrics and hasattr(latest_metrics, 'confusion_matrix') and latest_metrics.confusion_matrix:
        cm = latest_metrics.confusion_matrix
    elif latest_metrics:
        # Fallback to query if relationship isn't loaded
        cm = db.query(models.ConfusionMatrix).filter_by(run_id=latest_metrics.run_id).first()
        
    metrics = {
        'accuracy': latest_metrics.accuracy,
        'precision': latest_metrics.precision,
        'recall': latest_metrics.recall,
        'f1_score': latest_metrics.f1_score,
        'roc_auc': latest_metrics.roc_auc
    } if latest_metrics else None

    # Get model history for "Growth" chart
    models_history = db.query(models.ModelRegistry).order_by(models.ModelRegistry.created_at.desc()).limit(20).all()

    # Get Feature Importance for the latest run
    feature_importance = []
    if latest_metrics:
        feature_importance = db.query(models.FeatureImportance).filter_by(run_id=latest_metrics.run_id).order_by(models.FeatureImportance.importance_score.desc()).limit(10).all()

    return templates.TemplateResponse("analysis.html", {
        "request": request, 
        "metrics": metrics, 
        "cm": cm,
        "models": models_history,
        "features": feature_importance # Pass features to template
    })

@app.get("/smote-analysis", response_class=HTMLResponse)
async def smote_analysis_page(request: Request):
    return templates.TemplateResponse("smote_analysis.html", {"request": request})

@app.get("/history", response_class=HTMLResponse)
async def history(request: Request, db: Session = Depends(get_db)):
    predictions = db.query(models.Prediction).order_by(models.Prediction.created_at.desc()).limit(100).all()
    return templates.TemplateResponse("history.html", {"request": request, "predictions": predictions})

@app.post("/predict")
async def predict(
    request: Request,
    customer_id: str = Form(None),
    gender: str = Form(...),
    senior_citizen: int = Form(...),
    partner: str = Form(...),
    dependents: str = Form(...),
    tenure: int = Form(...),
    phone_service: str = Form(...),
    multiple_lines: str = Form(...),
    internet_service: str = Form(...),
    online_security: str = Form(...),
    online_backup: str = Form(...),
    device_protection: str = Form(...),
    tech_support: str = Form(...),
    streaming_tv: str = Form(...),
    streaming_movies: str = Form(...),
    contract: str = Form(...),
    paperless_billing: str = Form(...),
    payment_method: str = Form(...),
    monthly_charges: float = Form(...),
    total_charges: float = Form(...)
):
    # Default to MANUAL_{timestamp} if not provided
    if not customer_id or customer_id.strip() == "":
        customer_id = f"MANUAL_{int(datetime.now().timestamp())}"
    
    input_data = {
        'customerID': customer_id,
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    try:
        pipeline = ChurnPipeline()
        result = pipeline.predict_one(input_data)
        
        # Save prediction to DB
        db = SessionLocal()
        try:
            # Upsert Customer Data
            customer = db.query(models.Customer).filter_by(customer_id=customer_id).first()
            if not customer:
                customer = models.Customer(
                    customer_id=customer_id,
                    gender=gender,
                    senior_citizen=senior_citizen,
                    partner=partner,
                    dependents=dependents,
                    tenure=tenure,
                    phone_service=phone_service,
                    multiple_lines=multiple_lines,
                    internet_service=internet_service,
                    online_security=online_security,
                    online_backup=online_backup,
                    device_protection=device_protection,
                    tech_support=tech_support,
                    streaming_tv=streaming_tv,
                    streaming_movies=streaming_movies,
                    contract=contract,
                    paperless_billing=paperless_billing,
                    payment_method=payment_method,
                    monthly_charges=monthly_charges,
                    total_charges=total_charges,
                    churn="Unknown" # Since this is a new manual prediction
                )
                db.add(customer)
            else:
                # Update existing customer data
                customer.gender = gender
                customer.senior_citizen = senior_citizen
                customer.partner = partner
                customer.dependents = dependents
                customer.tenure = tenure
                customer.phone_service = phone_service
                customer.multiple_lines = multiple_lines
                customer.internet_service = internet_service
                customer.online_security = online_security
                customer.online_backup = online_backup
                customer.device_protection = device_protection
                customer.tech_support = tech_support
                customer.streaming_tv = streaming_tv
                customer.streaming_movies = streaming_movies
                customer.contract = contract
                customer.paperless_billing = paperless_billing
                customer.payment_method = payment_method
                customer.monthly_charges = monthly_charges
                customer.total_charges = total_charges
            
            db.commit()

            pred_entry = models.Prediction(
                customer_id=customer_id,
                churn_prediction=result['churn_prediction'],
                churn_probability=result['churn_probability'],
                risk_category=result['risk_category']
            )
            db.add(pred_entry)
            db.commit()
        except Exception as e:
            print(f"Error saving prediction: {e}")
            db.rollback()
        finally:
            db.close()
        
        strategy = get_retention_strategy(result['risk_category'], input_data)
        strategy_html = strategy.replace('\n', '<br>').replace('**', '<b>').replace('🚨', '').replace('⚠️', '').replace('✅', '')
        
        # --- AI Recommendation ---
        ai_recommendation = "AI recommendation unavailable."
        try:
            # Create a localized version of input data for AI context (IDR currency)
            input_data_localized = input_data.copy()
            exchange_rate = 15000
            try:
                mc = float(input_data.get('MonthlyCharges', 0))
                tc = float(input_data.get('TotalCharges', 0))
                input_data_localized['MonthlyCharges'] = f"Rp {int(mc * exchange_rate):,}".replace(",", ".")
                input_data_localized['TotalCharges'] = f"Rp {int(tc * exchange_rate):,}".replace(",", ".")
            except:
                pass

            # Prepare prompt
            prompt = f"""
            Sebagai Ahli Strategi Retensi Pelanggan, berikan rekomendasi langsung dan spesifik berdasarkan data ini.
            
            Profil Pelanggan:
            {json.dumps(input_data_localized, indent=2)}
            
            Prediksi Churn: {result['churn_prediction']} (Probabilitas: {result['churn_probability']:.1%})
            
            Berikan jawaban langsung tanpa pembuka, dalam format HTML:
            <h3>1. Analisis Situasi</h3>
            [Jelaskan mengapa dia berisiko/aman dalam 2 kalimat]
            
            <h3>2. Penawaran Retensi Spesifik</h3>
            [Sebutkan 2-3 langkah konkret atau penawaran diskon/layanan yang pas]
            
            <h3>3. Panduan Komunikasi</h3>
            [Contoh kalimat langsung untuk CS saat menghubungi pelanggan ini]
            """
            
            # Gemini 2.5 Flash Configuration (Direct Google API)
            api_key = os.getenv("GEMINI_API_KEY")
            model_name = os.getenv("AI_MODEL", "gemini-2.5-flash")
            
            if not api_key or "INSERT_YOUR" in api_key:
                ai_recommendation = "Mohon set GEMINI_API_KEY di file .env untuk mengaktifkan fitur AI."
            else:
                api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
                
                print(f"DEBUG: Calling Gemini API: {model_name}")
                
                # Gemini REST API payload structure
                payload = {
                    "contents": [{
                        "parts": [{"text": prompt}]
                    }]
                }
                
                response = requests.post(
                    api_url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    ai_data = response.json()
                    # Extract text from Gemini response
                    try:
                        ai_content = ai_data['candidates'][0]['content']['parts'][0]['text']
                        ai_recommendation = ai_content.replace('\n', '<br>').replace('**', '<b>')
                    except (KeyError, IndexError):
                        ai_recommendation = "Format respons AI tidak dikenali."
                else:
                    print(f"DEBUG: AI Error Status {response.status_code}: {response.text}")
                    ai_recommendation = f"Gagal menghubungi AI: {response.status_code} - {response.text}"

                    
        except Exception as ai_err:
            import traceback
            traceback.print_exc()
            print(f"DEBUG: Exception: {str(ai_err)}")
            ai_recommendation = f"AI Error: {str(ai_err)}"
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": result,
            "strategy": strategy_html,
            "ai_recommendation": ai_recommendation,
            "input": input_data
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": str(e)})

@app.get("/history/detail/{prediction_id}", response_class=HTMLResponse)
async def prediction_detail(request: Request, prediction_id: int, db: Session = Depends(get_db)):
    prediction = db.query(models.Prediction).filter(models.Prediction.prediction_id == prediction_id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    customer = db.query(models.Customer).filter(models.Customer.customer_id == prediction.customer_id).first()
    
    # Calculate Risk Factors (Simple Rule-Based Transparency)
    risk_factors = []
    if customer:
        if customer.contract == "Month-to-month":
            risk_factors.append({"name": "Kontrak Bulanan", "impact": "High", "desc": "Pelanggan tanpa kontrak jangka panjang lebih mudah pindah."})
        if customer.internet_service == "Fiber optic":
            risk_factors.append({"name": "Layanan Fiber Optic", "impact": "Medium", "desc": "Tingkat komplain pada layanan ini cenderung lebih tinggi."})
        if customer.payment_method == "Electronic check":
            risk_factors.append({"name": "Electronic Check", "impact": "Medium", "desc": "Metode pembayaran ini berkorelasi dengan churn tinggi."})
        if customer.tenure < 12:
            risk_factors.append({"name": "Pelanggan Baru", "impact": "High", "desc": "Pelanggan di tahun pertama masih dalam fase evaluasi."})
        if customer.monthly_charges > 80:
            risk_factors.append({"name": "Tagihan Tinggi", "impact": "Medium", "desc": "Biaya bulanan di atas rata-rata pasar."})
            
    return templates.TemplateResponse("prediction_detail.html", {
        "request": request, 
        "prediction": prediction, 
        "customer": customer,
        "risk_factors": risk_factors
    })

@app.get("/api/datasets")
async def list_datasets():
    datasets_dir = "datasets"
    if not os.path.exists(datasets_dir):
        return []
    
    files = []
    for f in os.listdir(datasets_dir):
        fp = os.path.join(datasets_dir, f)
        if os.path.isfile(fp):
            files.append({
                "filename": f,
                "size": os.path.getsize(fp),
                "created_at": datetime.fromtimestamp(os.path.getctime(fp)).strftime('%Y-%m-%d %H:%M:%S')
            })
    return sorted(files, key=lambda x: x['created_at'], reverse=True)

@app.get("/api/datasets/{filename}")
async def get_dataset_preview(filename: str):
    file_path = os.path.join("datasets", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
            
        # Preview first 10 rows
        preview = df.head(10).fillna("").to_dict(orient='records')
        columns = df.columns.tolist()
        return {"filename": filename, "columns": columns, "data": preview}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/datasets/{filename}")
async def delete_dataset(filename: str):
    file_path = os.path.join("datasets", filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"message": "File deleted"}
    raise HTTPException(status_code=404, detail="File not found")

@app.post("/upload_csv")
async def upload_csv(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        contents = await file.read()
        filename = file.filename
        
        # Save file to datasets folder
        datasets_dir = "datasets"
        if not os.path.exists(datasets_dir):
            os.makedirs(datasets_dir)
            
        file_path = os.path.join(datasets_dir, filename)
        with open(file_path, "wb") as f:
            f.write(contents)
            
        file_stream = io.BytesIO(contents)
        
        df = None
        if filename.lower().endswith('.csv'):
            try:
                df = pd.read_csv(file_stream, encoding='utf-8')
            except UnicodeDecodeError:
                file_stream.seek(0)
                df = pd.read_csv(file_stream, encoding='latin-1')
        elif filename.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_stream)
        else:
            return templates.TemplateResponse("upload.html", {
                "request": request, 
                "error": "Format file tidak didukung. Harap unggah CSV atau Excel (.xlsx)."
            })
        
        required_cols = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                         'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 
                         'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                         'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 
                         'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']
        
        # Normalize column names to match requirements (case insensitive check could be better, but strict for now)
        # Check if all required columns exist
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
             return templates.TemplateResponse("upload.html", {
                 "request": request, 
                 "error": f"Kolom hilang: {', '.join(missing_cols)}"
             })

        # Helper for robust float conversion
        def safe_float(val):
            try:
                if pd.isna(val) or str(val).strip() == "":
                    return 0.0
                return float(val)
            except (ValueError, TypeError):
                # Log or handle specific cases if needed, e.g. datetime
                return 0.0
        
        # Mapping CSV column names to Database model fields
        # This handles potential case sensitivity issues
        col_map = {
            'customerID': 'customer_id',
            'gender': 'gender',
            'SeniorCitizen': 'senior_citizen',
            'Partner': 'partner',
            'Dependents': 'dependents',
            'tenure': 'tenure',
            'PhoneService': 'phone_service',
            'MultipleLines': 'multiple_lines',
            'InternetService': 'internet_service',
            'OnlineSecurity': 'online_security',
            'OnlineBackup': 'online_backup',
            'DeviceProtection': 'device_protection',
            'TechSupport': 'tech_support',
            'StreamingTV': 'streaming_tv',
            'StreamingMovies': 'streaming_movies',
            'Contract': 'contract',
            'PaperlessBilling': 'paperless_billing',
            'PaymentMethod': 'payment_method',
            'MonthlyCharges': 'monthly_charges',
            'TotalCharges': 'total_charges',
            'Churn': 'churn'
        }

        added_count = 0
        for i, row in df.iterrows():
            tc = safe_float(row.get('TotalCharges'))
            mc = safe_float(row.get('MonthlyCharges'))
            
            # Use get() with default values to avoid KeyError
            customer_data = {
                'customer_id': str(row.get('customerID', f"UNKNOWN_{i}")),
                'gender': row.get('gender'),
                'senior_citizen': int(row.get('SeniorCitizen', 0)),
                'partner': row.get('Partner'),
                'dependents': row.get('Dependents'),
                'tenure': int(row.get('tenure', 0)),
                'phone_service': row.get('PhoneService'),
                'multiple_lines': row.get('MultipleLines'),
                'internet_service': row.get('InternetService'),
                'online_security': row.get('OnlineSecurity'),
                'online_backup': row.get('OnlineBackup'),
                'device_protection': row.get('DeviceProtection'),
                'tech_support': row.get('TechSupport'),
                'streaming_tv': row.get('StreamingTV'),
                'streaming_movies': row.get('StreamingMovies'),
                'contract': row.get('Contract'),
                'paperless_billing': row.get('PaperlessBilling'),
                'payment_method': row.get('PaymentMethod'),
                'monthly_charges': mc,
                'total_charges': tc,
                'churn': row.get('Churn')
            }
                
            customer = models.Customer(**customer_data)
            
            try:
                # Upsert: Check if exists, if so update, else insert
                existing = db.query(models.Customer).filter_by(customer_id=customer.customer_id).first()
                if not existing:
                    db.add(customer)
                    added_count += 1
                else:
                    # Update existing record
                    for key, value in customer_data.items():
                        setattr(existing, key, value)
            except:
                db.rollback()
                continue
        
        db.commit()
        return templates.TemplateResponse("upload.html", {
            "request": request, 
            "success": f"Berhasil mengimpor {added_count} baris data baru."
        })
        
    except Exception as e:
        return templates.TemplateResponse("upload.html", {"request": request, "error": f"Gagal memproses file: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
