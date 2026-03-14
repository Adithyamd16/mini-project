from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
import json

app = Flask(__name__)
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
db_path = os.path.join(os.path.dirname(__file__), 'Salary_Data.csv')
metrics_path = os.path.join(os.path.dirname(__file__), 'metrics.json')

@app.route('/')
def home():
    points = []
    metrics = None
    
    # Load EDA points
    try:
        if os.path.exists(db_path):
            data = pd.read_csv(db_path)
            points = [{'x': row['YearsExperience'], 'y': row['Salary']} for _, row in data.iterrows()]
    except Exception as e:
        print("Error loading CSV:", e)
        
    # Load Evaluation metrics
    try:
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
    except Exception as e:
        print("Error loading Metrics:", e)
        
    return render_template('index.html', chart_data=points, metrics=metrics)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        experience = float(data.get('experience', 0))
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            prediction = model.predict([[experience]])[0]
        else:
            prediction = 23700 + (experience * 9350)
            
        return jsonify({'success': True, 'prediction': prediction})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
