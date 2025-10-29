# ==============================
# SLIMNESS PREDICTION & HEALTH ANALYSIS WEB APP
# ==============================

# Install dependencies
!pip install flask==2.3.3 pyngrok==5.0.0 pandas scikit-learn numpy --quiet
!pkill ngrok

# Set ngrok auth token
!ngrok config add-authtoken 34edZqmWYS76T3OqbIj5OKrR22t_3KxNdFM7f18xhFXkjyHW

import os
import sys
from flask import Flask, render_template_string, request, jsonify
from pyngrok import ngrok
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Create Flask app
app = Flask(__name__)

# Global variables
rf_multi = None
scaler = None
label_encoders = {}
target_encoder = None
feature_columns = []

# Enhanced HTML Template for Slimness Prediction
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Slimness Prediction & Health Analysis</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        h1 {
            color: #2d3436;
            margin-bottom: 10px;
            font-size: 2.5em;
            background: linear-gradient(135deg, #0984e3, #00b894);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            color: #636e72;
            font-size: 1.2em;
            margin-bottom: 20px;
        }
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        .form-group.full-width {
            grid-column: 1 / -1;
        }
        label {
            font-weight: 600;
            color: #2d3436;
            margin-bottom: 8px;
            font-size: 0.95em;
        }
        input, select {
            padding: 12px;
            border: 2px solid #dfe6e9;
            border-radius: 10px;
            font-size: 1em;
            transition: all 0.3s;
            background: #f8f9fa;
        }
        input:focus, select:focus {
            outline: none;
            border-color: #74b9ff;
            box-shadow: 0 0 0 3px rgba(116, 185, 255, 0.1);
            background: white;
        }
        button {
            grid-column: 1 / -1;
            background: linear-gradient(135deg, #00b894 0%, #0984e3 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 12px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            margin-top: 20px;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(0, 184, 148, 0.4);
        }
        
        /* Results Styling */
        .result {
            margin-top: 30px;
            padding: 0;
            border-radius: 15px;
            overflow: hidden;
            display: none;
        }
        .result.show { display: block; }
        .result-header {
            background: linear-gradient(135deg, #00b894 0%, #0984e3 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        .result-content {
            padding: 25px;
            background: #f8f9fa;
        }
        .result-card {
            background: white;
            padding: 20px;
            margin: 15px 0;
            border-radius: 12px;
            border-left: 5px solid #74b9ff;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .result-card.warning { border-left-color: #e17055; }
        .result-card.success { border-left-color: #00b894; }
        .result-card.info { border-left-color: #74b9ff; }
        
        .feature-list {
            margin: 15px 0;
        }
        .feature-item {
            padding: 10px;
            margin: 8px 0;
            background: #f1f2f6;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .suggestions {
            background: #fff3cd;
            border-left: 5px solid #ffc107;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
        }
        .suggestions h3 {
            color: #856404;
            margin-bottom: 15px;
        }
        .suggestions ul {
            margin-left: 20px;
        }
        .suggestions li {
            margin: 10px 0;
            color: #856404;
        }
        
        .loading {
            text-align: center;
            padding: 30px;
            display: none;
        }
        .loading.show { display: block; }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #00b894;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @media (max-width: 768px) {
            .form-grid { grid-template-columns: 1fr; }
            h1 { font-size: 2em; }
            .container { padding: 20px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏃‍♂️ Slimness Prediction & Health Analysis</h1>
            <p class="subtitle">AI-Powered Analysis of Your Weight Status with Personalized Health Solutions</p>
        </div>
       
        <form id="predictionForm">
            <div class="form-grid">
                <!-- Basic Information -->
                <div class="form-group">
                    <label for="height">📏 Height (meters)</label>
                    <input type="number" id="height" step="0.01" required min="1.0" max="2.5" value="1.75" placeholder="e.g., 1.75">
                </div>
               
                <div class="form-group">
                    <label for="weight">⚖️ Weight (kg)</label>
                    <input type="number" id="weight" step="0.1" required min="30" max="200" value="68" placeholder="e.g., 68">
                </div>
               
                <div class="form-group">
                    <label for="age">🎂 Age</label>
                    <input type="number" id="age" required min="15" max="100" value="28" placeholder="e.g., 28">
                </div>
               
                <div class="form-group">
                    <label for="gender">👤 Gender</label>
                    <select id="gender" required>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                    </select>
                </div>
               
                <!-- Lifestyle Information -->
                <div class="form-group">
                    <label for="activity">💪 Physical Activity Level</label>
                    <select id="activity" required>
                        <option value="Sedentary">Sedentary (Little to no exercise)</option>
                        <option value="Light">Light (Light exercise 1-3 days/week)</option>
                        <option value="Moderate" selected>Moderate (Moderate exercise 3-5 days/week)</option>
                        <option value="High">High (Intense exercise 6-7 days/week)</option>
                    </select>
                </div>
               
                <div class="form-group">
                    <label for="steps">🚶‍♂️ Daily Steps</label>
                    <input type="number" id="steps" required min="0" max="50000" value="8500" placeholder="e.g., 8500">
                </div>
               
                <div class="form-group">
                    <label for="screenTime">📱 Screen Time (hours/day)</label>
                    <input type="number" id="screenTime" step="0.5" required min="0" max="24" value="5" placeholder="e.g., 5">
                </div>
               
                <!-- Nutrition Information -->
                <div class="form-group">
                    <label for="protein">🥩 Protein Intake (g/day)</label>
                    <input type="number" id="protein" required min="0" max="300" value="75" placeholder="e.g., 75">
                </div>
               
                <div class="form-group">
                    <label for="water">💧 Water Intake (liters/day)</label>
                    <input type="number" id="water" step="0.1" required min="0" max="10" value="2.5" placeholder="e.g., 2.5">
                </div>
               
                <div class="form-group">
                    <label for="highCalorie">🍔 High-Calorie Food Consumption</label>
                    <select id="highCalorie" required>
                        <option value="no" selected>Rarely</option>
                        <option value="yes">Frequently</option>
                    </select>
                </div>
               
                <div class="form-group">
                    <label for="vegetables">🥦 Vegetable Consumption</label>
                    <select id="vegetables" required>
                        <option value="yes" selected>Regularly</option>
                        <option value="no">Rarely</option>
                    </select>
                </div>
               
                <!-- Health Metrics -->
                <div class="form-group">
                    <label for="sleep">😴 Sleep Hours/Night</label>
                    <input type="number" id="sleep" step="0.5" required min="0" max="24" value="7.5" placeholder="e.g., 7.5">
                </div>
               
                <div class="form-group">
                    <label for="sleepQuality">⭐ Sleep Quality (1-10)</label>
                    <input type="number" id="sleepQuality" required min="1" max="10" value="7" placeholder="e.g., 7">
                </div>
               
                <div class="form-group">
                    <label for="stress">🧠 Stress Level (1-10)</label>
                    <input type="number" id="stress" required min="1" max="10" value="4" placeholder="e.g., 4">
                </div>
               
                <button type="submit">
                    🔍 Analyze My Health Status
                </button>
            </div>
        </form>
       
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p style="margin-top: 15px; color: #0984e3; font-weight: 600;">
                Analyzing your health profile and generating personalized recommendations...
            </p>
        </div>
       
        <div class="result" id="result"></div>
    </div>
   
    <script>
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
           
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
           
            loading.classList.add('show');
            result.classList.remove('show');
           
            const formData = {
                height: parseFloat(document.getElementById('height').value),
                weight: parseFloat(document.getElementById('weight').value),
                age: parseInt(document.getElementById('age').value),
                gender: document.getElementById('gender').value,
                activity: document.getElementById('activity').value,
                highCalorie: document.getElementById('highCalorie').value,
                vegetables: document.getElementById('vegetables').value,
                water: parseFloat(document.getElementById('water').value),
                sleep: parseFloat(document.getElementById('sleep').value),
                sleepQuality: parseInt(document.getElementById('sleepQuality').value),
                screenTime: parseFloat(document.getElementById('screenTime').value),
                steps: parseInt(document.getElementById('steps').value),
                protein: parseInt(document.getElementById('protein').value),
                stress: parseInt(document.getElementById('stress').value)
            };
           
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(formData)
                });
               
                const data = await response.json();
               
                loading.classList.remove('show');
               
                if (data.error) {
                    result.innerHTML = `
                        <div class="result-header">
                            <h2>❌ Error</h2>
                        </div>
                        <div class="result-content">
                            <div class="result-card warning">
                                <p>${data.error}</p>
                            </div>
                        </div>
                    `;
                } else {
                    // Determine card color based on category
                    const category = data.category.toLowerCase();
                    let cardClass = 'info';
                    if (category.includes('underweight')) cardClass = 'warning';
                    if (category.includes('healthy')) cardClass = 'success';
                    if (category.includes('obese')) cardClass = 'warning';
                    
                    result.innerHTML = `
                        <div class="result-header">
                            <h2>📊 Your Health Analysis Report</h2>
                            <p style="margin-top: 10px; opacity: 0.9;">Comprehensive analysis based on your input data</p>
                        </div>
                       
                        <div class="result-content">
                            <!-- Main Prediction Card -->
                            <div class="result-card ${cardClass}">
                                <h3 style="color: #2d3436; margin-bottom: 15px;">🎯 Weight Status Prediction</h3>
                                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                                    <div>
                                        <strong>Predicted Category:</strong><br>
                                        <span style="font-size: 1.2em; color: #0984e3;">${data.category}</span>
                                    </div>
                                    <div>
                                        <strong>Confidence Score:</strong><br>
                                        <span style="font-size: 1.2em; color: #00b894;">${(data.confidence * 100).toFixed(1)}%</span>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- BMI Information -->
                            <div class="result-card">
                                <h3 style="color: #2d3436; margin-bottom: 15px;">⚖️ Body Mass Index (BMI) Analysis</h3>
                                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                                    <div>
                                        <strong>Your BMI:</strong><br>
                                        <span style="font-size: 1.3em; font-weight: bold;">${data.bmi.toFixed(1)}</span>
                                    </div>
                                    <div>
                                        <strong>Classification:</strong><br>
                                        <span style="font-size: 1.1em; color: ${data.bmi_category === 'Normal weight' ? '#00b894' : '#e17055'}">
                                            ${data.bmi_category}
                                        </span>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Key Influencing Factors -->
                            <div class="result-card">
                                <h3 style="color: #2d3436; margin-bottom: 15px;">🔍 Top Influencing Factors</h3>
                                <p style="margin-bottom: 15px; color: #636e72;">These factors had the most impact on your prediction:</p>
                                <div class="feature-list">
                                    ${data.top_features.map((f, index) => `
                                        <div class="feature-item">
                                            <div>
                                                <strong>${f.feature.replace(/_/g, ' ').replace(/([A-Z])/g, ' $1')}:</strong>
                                                ${f.value}
                                            </div>
                                            <div style="color: #0984e3; font-weight: 600;">
                                                ${(f.importance * 100).toFixed(1)}%
                                            </div>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                            
                            <!-- Detailed Reasons Analysis -->
                            <div class="result-card">
                                <h3 style="color: #2d3436; margin-bottom: 15px;">🔬 Reasons Analysis</h3>
                                <div style="line-height: 1.6;">
                                    ${data.reasons.map(reason => `<p style="margin: 10px 0;">• ${reason}</p>`).join('')}
                                </div>
                            </div>
                            
                            <!-- Personalized Recommendations -->
                            ${data.suggestions.length > 0 ? `
                            <div class="suggestions">
                                <h3>💡 Personalized Health Recommendations</h3>
                                <ul>
                                    ${data.suggestions.map(s => `<li>${s}</li>`).join('')}
                                </ul>
                            </div>
                            ` : ''}
                            
                            <!-- Action Plan -->
                            <div class="result-card success">
                                <h3 style="color: #2d3436; margin-bottom: 15px;">📅 Recommended Action Plan</h3>
                                <div style="line-height: 1.6;">
                                    ${data.action_plan.map(step => `<p style="margin: 8px 0;">✅ ${step}</p>`).join('')}
                                </div>
                            </div>
                        </div>
                    `;
                }
               
                result.classList.add('show');
                result.scrollIntoView({ behavior: 'smooth', block: 'start' });
               
            } catch (error) {
                loading.classList.remove('show');
                result.innerHTML = `
                    <div class="result-header">
                        <h2>❌ Connection Error</h2>
                    </div>
                    <div class="result-content">
                        <div class="result-card warning">
                            <p>Failed to connect to analysis service: ${error.message}</p>
                            <p style="margin-top: 10px;">Please check your connection and try again.</p>
                        </div>
                    </div>
                `;
                result.classList.add('show');
            }
        });
    </script>
</body>
</html>
"""

# Function to train the model
def train_model():
    global rf_multi, scaler, label_encoders, target_encoder, feature_columns
    
    print("📊 Loading and preparing dataset...")
    # Load your dataset
    df = pd.read_csv('augmented_obesity_lifestyle_dataset (1).csv')
    
    print("🔧 Preprocessing data...")
    
    # Handle categorical variables
    categorical_columns = ['Gender', 'PhysicalActivity', 'FrequentConsumptionHighCalorieFood', 'FrequentVegetableConsumption']
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Encode target
    target_encoder = LabelEncoder()
    df['Category_encoded'] = target_encoder.fit_transform(df['Category'])
    
    # Select features
    feature_columns = [
        'Height_m', 'Weight_kg', 'Age', 'Gender', 'PhysicalActivity',
        'FrequentConsumptionHighCalorieFood', 'FrequentVegetableConsumption',
        'BMI', 'Water_Intake_L', 'Sleep_Hours', 'Sleep_Quality_Score',
        'Screen_Time_Hours', 'Steps_Per_Day', 'Protein_Intake_g', 'Stress_Level_Score'
    ]
    
    X = df[feature_columns]
    y = df['Category_encoded']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    print("🤖 Training Random Forest model...")
    rf_multi = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_multi.fit(X_train_scaled, y_train)
    
    # Calculate accuracy
    from sklearn.metrics import accuracy_score
    y_pred = rf_multi.predict(scaler.transform(X_test))
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Model trained successfully! Accuracy: {accuracy:.4f}")

# Function to analyze reasons for prediction
def analyze_reasons(category, user_data, top_features):
    reasons = []
    bmi = user_data['bmi']
    
    if category == 'Underweight':
        reasons.append(f"Low BMI ({bmi:.1f}) indicates underweight condition")
        if user_data['protein'] < 50:
            reasons.append(f"Insufficient protein intake ({user_data['protein']}g) for muscle maintenance")
        if user_data['steps'] > 10000:
            reasons.append("High activity level without adequate nutritional support")
        if user_data['highCalorie'] == 'no':
            reasons.append("Limited consumption of calorie-dense foods")
            
    elif category == 'Healthy Slim':
        reasons.append(f"Optimal BMI ({bmi:.1f}) within healthy range")
        if user_data['steps'] >= 8000:
            reasons.append("Good daily activity level supporting healthy metabolism")
        if user_data['vegetables'] == 'yes':
            reasons.append("Regular vegetable consumption contributing to balanced nutrition")
        if user_data['sleep'] >= 7:
            reasons.append("Adequate sleep supporting overall health")
            
    elif category == 'Overweight':
        reasons.append(f"Elevated BMI ({bmi:.1f}) indicating excess weight")
        if user_data['steps'] < 5000:
            reasons.append("Low daily activity level affecting calorie expenditure")
        if user_data['highCalorie'] == 'yes':
            reasons.append("Frequent high-calorie food consumption")
        if user_data['screenTime'] > 6:
            reasons.append("High screen time potentially reducing physical activity")
            
    elif category == 'Obese':
        reasons.append(f"High BMI ({bmi:.1f}) indicating obesity")
        if user_data['steps'] < 4000:
            reasons.append("Very low daily activity level")
        if user_data['protein'] > 100 and user_data['highCalorie'] == 'yes':
            reasons.append("High calorie intake with potential nutritional imbalance")
        if user_data['sleep'] < 6:
            reasons.append("Inadequate sleep affecting metabolism and appetite regulation")
    
    # Add feature-specific reasons
    for feature in top_features[:2]:
        feature_name = feature['feature']
        value = feature['value']
        
        if feature_name == 'PhysicalActivity' and value in [0, 1]: # Sedentary/Light
            reasons.append("Low physical activity level impacting energy balance")
        elif feature_name == 'Protein_Intake_g' and value < 50:
            reasons.append("Suboptimal protein intake for muscle maintenance")
        elif feature_name == 'Steps_Per_Day' and value < 6000:
            reasons.append("Insufficient daily steps for cardiovascular health")
        elif feature_name == 'Screen_Time_Hours' and value > 6:
            reasons.append("Excessive screen time affecting activity patterns")
    
    return reasons

# Function to get personalized suggestions
def get_suggestions(category, user_data):
    suggestions = []
    action_plan = []
    
    bmi = user_data['bmi']
    steps = user_data['steps']
    sleep = user_data['sleep']
    water = user_data['water']
    stress = user_data['stress']
    protein = user_data['protein']
    screen_time = user_data['screenTime']
    activity = user_data['activity']
    
    if category in ['Underweight', 'Healthy Slim']:
        if bmi < 18.5:
            suggestions.append("Increase daily calorie intake by 300-500 calories with nutrient-dense foods")
            suggestions.append("Focus on protein-rich meals (1.2-1.6g per kg of body weight)")
            suggestions.append("Incorporate strength training 2-3 times per week")
            suggestions.append("Eat smaller, more frequent meals throughout the day")
            
            action_plan.extend([
                "Week 1-2: Increase protein intake to 70-80g daily",
                "Week 3-4: Add 2 strength training sessions weekly",
                "Week 5-6: Implement 5-6 smaller meals throughout day",
                "Week 7-8: Monitor weight gain and adjust calories"
            ])
        else:
            suggestions.append("Maintain current balanced diet with proper portion control")
            suggestions.append("Continue regular physical activity routine")
            suggestions.append("Monitor weight monthly to maintain stability")
            suggestions.append("Focus on nutrient timing around workouts")
            
            action_plan.extend([
                "Maintain current exercise routine",
                "Continue balanced nutrition pattern",
                "Monthly weight and measurement checks",
                "Adjust intake based on activity changes"
            ])
    
    # Activity recommendations
    if steps < 5000:
        suggestions.append(f"Gradually increase daily steps from {steps} to 8,000-10,000")
        action_plan.append(f"Increase steps by 1,000 weekly until reaching 8,000")
    
    if screen_time > 6:
        suggestions.append(f"Reduce recreational screen time from {screen_time} to under 4 hours")
        action_plan.append("Implement screen-free time 1 hour before bed")
    
    if sleep < 7:
        suggestions.append(f"Increase sleep duration from {sleep} to 7-9 hours nightly")
        action_plan.append("Establish consistent bedtime routine")
    
    if water < 2:
        suggestions.append(f"Increase water intake from {water}L to 2-3L daily")
        action_plan.append("Carry water bottle and set hourly reminders")
    
    if stress > 6:
        suggestions.append("Practice stress management techniques daily")
        action_plan.append("10-minute meditation or deep breathing daily")
    
    if protein < 60 and category in ['Underweight', 'Healthy Slim']:
        suggestions.append(f"Increase protein intake from {protein}g to 70-90g daily")
        action_plan.append("Add protein source to each meal and snack")
    
    return suggestions, action_plan

# Routes
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Calculate BMI
        bmi = data['weight'] / (data['height'] ** 2)
        
        # Prepare user input
        user_input = []
        for col in feature_columns:
            if col == 'BMI':
                user_input.append(bmi)
            elif col == 'Height_m':
                user_input.append(data['height'])
            elif col == 'Weight_kg':
                user_input.append(data['weight'])
            elif col == 'Age':
                user_input.append(data['age'])
            elif col == 'Gender':
                user_input.append(label_encoders['Gender'].transform([data['gender']])[0])
            elif col == 'PhysicalActivity':
                user_input.append(label_encoders['PhysicalActivity'].transform([data['activity']])[0])
            elif col == 'FrequentConsumptionHighCalorieFood':
                user_input.append(label_encoders['FrequentConsumptionHighCalorieFood'].transform([data['highCalorie']])[0])
            elif col == 'FrequentVegetableConsumption':
                user_input.append(label_encoders['FrequentVegetableConsumption'].transform([data['vegetables']])[0])
            elif col == 'Water_Intake_L':
                user_input.append(data['water'])
            elif col == 'Sleep_Hours':
                user_input.append(data['sleep'])
            elif col == 'Sleep_Quality_Score':
                user_input.append(data['sleepQuality'])
            elif col == 'Screen_Time_Hours':
                user_input.append(data['screenTime'])
            elif col == 'Steps_Per_Day':
                user_input.append(data['steps'])
            elif col == 'Protein_Intake_g':
                user_input.append(data['protein'])
            elif col == 'Stress_Level_Score':
                user_input.append(data['stress'])
        
        # Scale input
        user_input_scaled = scaler.transform([user_input])
        
        # Predict
        prediction = rf_multi.predict(user_input_scaled)[0]
        probabilities = rf_multi.predict_proba(user_input_scaled)[0]
        
        # Get category
        category = target_encoder.inverse_transform([prediction])[0]
        confidence = probabilities[prediction]
        
        # BMI category
        if bmi < 18.5:
            bmi_category = "Underweight"
        elif 18.5 <= bmi < 25:
            bmi_category = "Normal weight"
        elif 25 <= bmi < 30:
            bmi_category = "Overweight"
        else:
            bmi_category = "Obese"
        
        # Top features
        importances = rf_multi.feature_importances_
        top_indices = np.argsort(importances)[-3:][::-1]
        top_features = []
        for idx in top_indices:
            top_features.append({
                'feature': feature_columns[idx],
                'value': user_input[idx],
                'importance': importances[idx]
            })
        
        # Analyze reasons
        user_data_with_bmi = data.copy()
        user_data_with_bmi['bmi'] = bmi
        reasons = analyze_reasons(category, user_data_with_bmi, top_features)
        
        # Get suggestions and action plan
        suggestions, action_plan = get_suggestions(category, user_data_with_bmi)
        
        return jsonify({
            'category': category,
            'confidence': float(confidence),
            'bmi': float(bmi),
            'bmi_category': bmi_category,
            'top_features': top_features,
            'reasons': reasons,
            'suggestions': suggestions,
            'action_plan': action_plan
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Main execution
if __name__ == '__main__':
    print("🚀 Starting Slimness Prediction Web Application...")
    print("="*60)
    
    # Train model
    train_model()
    
    # Start ngrok tunnel
    port = 5000
    public_url = ngrok.connect(port)
    
    print("\n" + "="*60)
    print("✅ SLIMNESS PREDICTION APP IS RUNNING!")
    print("="*60)
    print(f"🌐 Public URL: {public_url}")
    print(f"📍 Local URL: http://127.0.0.1:{port}")
    print("="*60)
    print("\n👉 Click the public URL above to access your app!")
    print("Press CTRL+C to stop the server\n")
    
    # Run Flask app
    app.run(port=port)
