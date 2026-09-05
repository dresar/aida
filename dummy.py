import pandas as pd
import random
import string
import os
from datetime import datetime

def generate_dummy_data(num_rows=1000):
    data = []
    
    # Options for categorical columns
    genders = ['Male', 'Female']
    binary = ['Yes', 'No']
    internet_services = ['DSL', 'Fiber optic', 'No']
    contracts = ['Month-to-month', 'One year', 'Two year']
    payment_methods = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
    
    for _ in range(num_rows):
        # Generate Customer ID (e.g., 1234-ABCD)
        cust_id_num = ''.join(random.choices(string.digits, k=4))
        cust_id_str = ''.join(random.choices(string.ascii_uppercase, k=5))
        customer_id = f"{cust_id_num}-{cust_id_str}"
        
        gender = random.choice(genders)
        senior_citizen = random.choice([0, 1])
        partner = random.choice(binary)
        dependents = random.choice(binary)
        tenure = random.randint(1, 72)
        phone_service = random.choice(binary)
        
        if phone_service == 'No':
            multiple_lines = 'No phone service'
        else:
            multiple_lines = random.choice(['Yes', 'No'])
            
        internet_service = random.choice(internet_services)
        
        if internet_service == 'No':
            online_security = 'No internet service'
            online_backup = 'No internet service'
            device_protection = 'No internet service'
            tech_support = 'No internet service'
            streaming_tv = 'No internet service'
            streaming_movies = 'No internet service'
        else:
            online_security = random.choice(binary)
            online_backup = random.choice(binary)
            device_protection = random.choice(binary)
            tech_support = random.choice(binary)
            streaming_tv = random.choice(binary)
            streaming_movies = random.choice(binary)
            
        contract = random.choice(contracts)
        paperless_billing = random.choice(binary)
        payment_method = random.choice(payment_methods)
        
        monthly_charges = round(random.uniform(18.25, 118.75), 2)
        total_charges = round(monthly_charges * tenure, 2)
        
        # Random Churn with some logic (higher charges/month-to-month -> higher churn chance)
        churn_prob = 0.1
        if contract == 'Month-to-month': churn_prob += 0.3
        if internet_service == 'Fiber optic': churn_prob += 0.1
        if total_charges > 2000: churn_prob -= 0.1
        
        churn = 'Yes' if random.random() < churn_prob else 'No'
        
        row = {
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
            'TotalCharges': total_charges,
            'Churn': churn
        }
        data.append(row)
        
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Ensure datasets directory exists
    output_dir = os.path.join(os.getcwd(), "datasets")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Generate data
    df = generate_dummy_data(1000)
    
    # Save to CSV with timestamp to be unique
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dummy_data_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    df.to_csv(filepath, index=False)
    print(f"Berhasil membuat 1000 data dummy di: {filepath}")
