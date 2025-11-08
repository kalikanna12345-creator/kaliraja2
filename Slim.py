# ==============================
# COMPLETE ENHANCED SLIMNESS PREDICTION WITH DATASET VISUALIZATION
# ==============================

# Install dependencies
!pip install flask==2.3.3 pyngrok==5.0.0 pandas scikit-learn numpy fpdf reportlab matplotlib seaborn --quiet

# IMPORTANT: Set your ngrok auth token here
NGROK_AUTH_TOKEN = "35CPeS8HuqUeP7yUlOMzLeQN1ar_3xk14KY9p6kCoHHJTBejC"

import os
import json
import datetime
import base64
from io import BytesIO
from flask import Flask, render_template_string, request, jsonify, send_file
from pyngrok import ngrok
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set ngrok auth token
if NGROK_AUTH_TOKEN != "YOUR_NGROK_AUTH_TOKEN_HERE":
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
else:
    print("\n⚠️  WARNING: NGROK AUTH TOKEN NOT SET!")
    exit()

# Create Flask app
app = Flask(__name__)

# Global variables
rf_multi = None
scaler = None
label_encoders = {}
target_encoder = None
feature_columns = []
dataset_df = None

# Data storage file
DATA_FILE = 'patient_records.json'

# Initialize data storage
def init_data_storage():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'w') as f:
            json.dump([], f)

# Load patient records
def load_patient_records():
    try:
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

# Save patient record
def save_patient_record(record):
    records = load_patient_records()
    records.append(record)
    with open(DATA_FILE, 'w') as f:
        json.dump(records, f, indent=2)

# Generate pie charts
def generate_pie_charts():
    global dataset_df

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Dataset Distribution Analysis', fontsize=20, fontweight='bold')

    # Read original dataset for proper labels
    df_original = pd.read_csv('augmented_obesity_lifestyle_dataset (1).csv')

    # 1. Category Distribution
    category_counts = df_original['Category'].value_counts()
    colors1 = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#ffa726']
    axes[0, 0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
                   startangle=90, colors=colors1, textprops={'fontsize': 11, 'weight': 'bold'})
    axes[0, 0].set_title('Category Distribution', fontsize=14, fontweight='bold')

    # 2. Gender Distribution
    gender_counts = df_original['Gender'].value_counts()
    colors2 = ['#667eea', '#764ba2']
    axes[0, 1].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%',
                   startangle=90, colors=colors2, textprops={'fontsize': 11, 'weight': 'bold'})
    axes[0, 1].set_title('Gender Distribution', fontsize=14, fontweight='bold')

    # 3. Physical Activity Distribution
    activity_counts = df_original['PhysicalActivity'].value_counts()
    colors3 = ['#ff6b6b', '#ffa726', '#66bb6a', '#42a5f5']
    axes[0, 2].pie(activity_counts.values, labels=activity_counts.index, autopct='%1.1f%%',
                   startangle=90, colors=colors3, textprops={'fontsize': 11, 'weight': 'bold'})
    axes[0, 2].set_title('Physical Activity Levels', fontsize=14, fontweight='bold')

    # 4. High Calorie Food Consumption
    calorie_counts = df_original['FrequentConsumptionHighCalorieFood'].value_counts()
    colors4 = ['#66bb6a', '#ff6b6b']
    axes[1, 0].pie(calorie_counts.values, labels=calorie_counts.index, autopct='%1.1f%%',
                   startangle=90, colors=colors4, textprops={'fontsize': 11, 'weight': 'bold'})
    axes[1, 0].set_title('High Calorie Food Consumption', fontsize=14, fontweight='bold')

    # 5. Vegetable Consumption
    veg_counts = df_original['FrequentVegetableConsumption'].value_counts()
    colors5 = ['#66bb6a', '#ffa726']
    axes[1, 1].pie(veg_counts.values, labels=veg_counts.index, autopct='%1.1f%%',
                   startangle=90, colors=colors5, textprops={'fontsize': 11, 'weight': 'bold'})
    axes[1, 1].set_title('Vegetable Consumption', fontsize=14, fontweight='bold')

    # 6. Age Groups
    age_bins = [0, 20, 30, 40, 50, 100]
    age_labels = ['0-20', '21-30', '31-40', '41-50', '50+']
    df_original['AgeGroup'] = pd.cut(df_original['Age'], bins=age_bins, labels=age_labels)
    age_group_counts = df_original['AgeGroup'].value_counts()
    colors6 = ['#ff6b6b', '#ffa726', '#66bb6a', '#42a5f5', '#ab47bc']
    axes[1, 2].pie(age_group_counts.values, labels=age_group_counts.index, autopct='%1.1f%%',
                   startangle=90, colors=colors6, textprops={'fontsize': 11, 'weight': 'bold'})
    axes[1, 2].set_title('Age Group Distribution', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()

    return image_base64

# Complete HTML Template
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
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 25px;
            box-shadow: 0 25px 80px rgba(0,0,0,0.3);
            padding: 40px;
            position: relative;
            overflow: hidden;
        }
        .container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 8px;
            background: linear-gradient(90deg, #ff6b6b, #ffa726, #66bb6a, #42a5f5, #ab47bc);
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        h1 {
            color: #2d3436;
            margin-bottom: 15px;
            font-size: 3em;
            background: linear-gradient(135deg, #ff6b6b, #ffa726, #66bb6a, #42a5f5, #ab47bc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .subtitle {
            color: #636e72;
            font-size: 1.3em;
            margin-bottom: 25px;
            font-weight: 300;
        }
        .nav-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            padding: 5px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .nav-tabs {
            display: flex;
            justify-content: space-between;
        }
        .nav-tab {
            flex: 1;
            padding: 18px 25px;
            text-align: center;
            cursor: pointer;
            border-radius: 12px;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            font-weight: 600;
            color: white;
            background: transparent;
            border: none;
            font-size: 1.1em;
        }
        .nav-tab:hover {
            background: rgba(255,255,255,0.15);
            transform: translateY(-3px);
        }
        .nav-tab.active {
            background: white;
            color: #667eea;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            transform: translateY(-2px);
        }
        .tab-content {
            display: none;
            animation: fadeIn 0.5s ease-in;
        }
        .tab-content.active {
            display: block;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
            margin-bottom: 25px;
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
            margin-bottom: 10px;
            font-size: 1em;
        }
        input, select {
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            font-size: 1em;
            transition: all 0.3s;
            background: #f8f9fa;
        }
        input:focus, select:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
            background: white;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 18px 35px;
            border: none;
            border-radius: 15px;
            font-size: 1.2em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            margin-top: 25px;
            width: 100%;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        }
        button:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
        }
        .loading {
            text-align: center;
            padding: 40px;
            display: none;
        }
        .loading.show { display: block; }
        .spinner {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #667eea;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            animation: spin 1s linear infinite;
            margin: 0 auto 25px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .result {
            margin-top: 30px;
            padding: 0;
            border-radius: 20px;
            overflow: hidden;
            display: none;
        }
        .result.show { display: block; }
        .result-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            text-align: center;
        }
        .result-content {
            padding: 30px;
            background: #f8f9fa;
        }
        .result-card {
            background: white;
            padding: 25px;
            margin: 20px 0;
            border-radius: 15px;
            border-left: 6px solid #667eea;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        .result-card h3 {
            color: #667eea;
            margin-bottom: 15px;
        }
        .result-card p {
            margin: 10px 0;
            line-height: 1.6;
        }
        .chart-container {
            background: white;
            padding: 30px;
            border-radius: 15px;
            margin: 20px 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        .chart-container img {
            width: 100%;
            border-radius: 10px;
        }
        .data-table {
            width: 100%;
            overflow-x: auto;
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }
        th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px;
            text-align: left;
            position: sticky;
            top: 0;
        }
        td {
            padding: 10px;
            border-bottom: 1px solid #e0e0e0;
        }
        tr:hover {
            background: #f8f9fa;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            text-align: center;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .stat-label {
            color: #636e72;
            margin-top: 10px;
        }
        @media (max-width: 768px) {
            .form-grid { grid-template-columns: 1fr; }
            .nav-tabs { flex-direction: column; }
            h1 { font-size: 2.2em; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏃‍♂️ Slimness Prediction</h1>
            <p class="subtitle">AI-Powered Health Assessment & Analytics</p>
        </div>

        <div class="nav-container">
            <div class="nav-tabs">
                <button class="nav-tab active" onclick="showTab('predict')">🔍 Prediction</button>
                <button class="nav-tab" onclick="showTab('visualize')">📊 Data Visualization</button>
                <button class="nav-tab" onclick="showTab('dataset')">📋 Dataset</button>
            </div>
        </div>

        <!-- Prediction Tab -->
        <div id="predict" class="tab-content active">
            <form id="predictionForm">
                <div class="form-grid">
                    <div class="form-group full-width">
                        <h3 style="color: #667eea;">👤 Patient Information</h3>
                    </div>
                    <div class="form-group">
                        <label>📛 Full Name</label>
                        <input type="text" id="name" required>
                    </div>
                    <div class="form-group">
                        <label>📧 Email</label>
                        <input type="email" id="email" required>
                    </div>
                    <div class="form-group">
                        <label>📞 Phone</label>
                        <input type="tel" id="phone" required>
                    </div>
                    <div class="form-group">
                        <label>📅 Date</label>
                        <input type="date" id="date" required>
                    </div>

                    <div class="form-group full-width">
                        <h3 style="color: #667eea;">📊 Health Metrics</h3>
                    </div>
                    <div class="form-group">
                        <label>📏 Height (m)</label>
                        <input type="number" id="height" step="0.01" value="1.75" required>
                    </div>
                    <div class="form-group">
                        <label>⚖️ Weight (kg)</label>
                        <input type="number" id="weight" step="0.1" value="68" required>
                    </div>
                    <div class="form-group">
                        <label>🎂 Age</label>
                        <input type="number" id="age" value="28" required>
                    </div>
                    <div class="form-group">
                        <label>👤 Gender</label>
                        <select id="gender" required>
                            <option value="Male">Male</option>
                            <option value="Female">Female</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>🏃 Physical Activity</label>
                        <select id="activity" required>
                            <option value="Sedentary">Sedentary</option>
                            <option value="Light">Light</option>
                            <option value="Moderate" selected>Moderate</option>
                            <option value="High">High</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>🚶 Daily Steps</label>
                        <input type="number" id="steps" value="8500" required>
                    </div>
                    <div class="form-group">
                        <label>📱 Screen Time (h)</label>
                        <input type="number" id="screenTime" step="0.5" value="5" required>
                    </div>
                    <div class="form-group">
                        <label>🥩 Protein (g/day)</label>
                        <input type="number" id="protein" value="75" required>
                    </div>
                    <div class="form-group">
                        <label>💧 Water (L/day)</label>
                        <input type="number" id="water" step="0.1" value="2.5" required>
                    </div>
                    <div class="form-group">
                        <label>🍔 High-Calorie Food</label>
                        <select id="highCalorie" required>
                            <option value="no" selected>Rarely</option>
                            <option value="yes">Frequently</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>🥦 Vegetables</label>
                        <select id="vegetables" required>
                            <option value="yes" selected>Regularly</option>
                            <option value="no">Rarely</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>😴 Sleep (hours)</label>
                        <input type="number" id="sleep" step="0.5" value="7.5" required>
                    </div>
                    <div class="form-group">
                        <label>⭐ Sleep Quality (1-10)</label>
                        <input type="number" id="sleepQuality" value="7" min="1" max="10" required>
                    </div>
                    <div class="form-group">
                        <label>🧠 Stress Level (1-10)</label>
                        <input type="number" id="stress" value="4" min="1" max="10" required>
                    </div>

                    <button type="submit" class="form-group full-width">
                        🔍 Analyze Health Status
                    </button>
                </div>
            </form>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="color: #667eea; font-weight: 600;">Analyzing your health profile...</p>
            </div>

            <div class="result" id="result"></div>
        </div>

        <!-- Visualization Tab -->
        <div id="visualize" class="tab-content">
            <h2 style="color: #667eea; margin-bottom: 20px;">📊 Dataset Distribution Analysis</h2>

            <div class="stats-grid" id="statsGrid"></div>

            <div class="chart-container">
                <img id="pieCharts" src="" alt="Loading charts..." />
            </div>
        </div>

        <!-- Dataset Tab -->
        <div id="dataset" class="tab-content">
            <h2 style="color: #667eea; margin-bottom: 20px;">📋 Training Dataset</h2>
            <div class="data-table" id="dataTable"></div>
        </div>
    </div>

    <script>
        document.getElementById('date').value = new Date().toISOString().split('T')[0];

        function showTab(tabName) {
            const tabs = document.querySelectorAll('.tab-content');
            const navTabs = document.querySelectorAll('.nav-tab');

            tabs.forEach(tab => tab.classList.remove('active'));
            navTabs.forEach(tab => tab.classList.remove('active'));

            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');

            if (tabName === 'visualize') {
                loadVisualizations();
            } else if (tabName === 'dataset') {
                loadDataset();
            }
        }

        async function loadVisualizations() {
            try {
                const response = await fetch('/get_visualizations');
                const data = await response.json();

                document.getElementById('pieCharts').src = 'data:image/png;base64,' + data.chart;

                const statsHtml = `
                    <div class="stat-card">
                        <div class="stat-value">${data.stats.total_records}</div>
                        <div class="stat-label">Total Records</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${data.stats.avg_age}</div>
                        <div class="stat-label">Average Age</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${data.stats.avg_bmi}</div>
                        <div class="stat-label">Average BMI</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${data.stats.male_percent}%</div>
                        <div class="stat-label">Male</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${data.stats.female_percent}%</div>
                        <div class="stat-label">Female</div>
                    </div>
                `;
                document.getElementById('statsGrid').innerHTML = statsHtml;
            } catch (error) {
                console.error('Error loading visualizations:', error);
            }
        }

        async function loadDataset() {
            try {
                const response = await fetch('/get_dataset');
                const data = await response.json();

                let tableHtml = '<table><thead><tr>';
                data.columns.forEach(col => {
                    tableHtml += `<th>${col}</th>`;
                });
                tableHtml += '</tr></thead><tbody>';

                data.rows.slice(0, 100).forEach(row => {
                    tableHtml += '<tr>';
                    row.forEach(cell => {
                        tableHtml += `<td>${cell}</td>`;
                    });
                    tableHtml += '</tr>';
                });
                tableHtml += '</tbody></table>';
                tableHtml += `<p style="text-align: center; margin-top: 20px; color: #636e72;">Showing first 100 of ${data.total_rows} records</p>`;

                document.getElementById('dataTable').innerHTML = tableHtml;
            } catch (error) {
                console.error('Error loading dataset:', error);
            }
        }

        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();

            const loading = document.getElementById('loading');
            const result = document.getElementById('result');

            loading.classList.add('show');
            result.classList.remove('show');

            const formData = {
                name: document.getElementById('name').value,
                email: document.getElementById('email').value,
                phone: document.getElementById('phone').value,
                date: document.getElementById('date').value,
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

                // Determine color scheme based on category
                let categoryColor = '#667eea';
                let categoryEmoji = '📊';
                if (data.category === 'Underweight') {
                    categoryColor = '#ff6b6b';
                    categoryEmoji = '⚠️';
                } else if (data.category === 'Healthy Slim') {
                    categoryColor = '#66bb6a';
                    categoryEmoji = '✅';
                } else if (data.category === 'Overweight') {
                    categoryColor = '#ffa726';
                    categoryEmoji = '⚡';
                } else if (data.category === 'Obese') {
                    categoryColor = '#e74c3c';
                    categoryEmoji = '🚨';
                }

                result.innerHTML = `
                    <div class="result-header" style="background: linear-gradient(135deg, ${categoryColor} 0%, ${categoryColor}dd 100%);">
                        <h2>${categoryEmoji} Health Analysis Report</h2>
                        <p style="font-size: 1.1em; margin-top: 10px;">${formData.name} - ${formData.date}</p>
                    </div>
                    <div class="result-content">
                        <div class="result-card" style="border-left-color: ${categoryColor};">
                            <h3 style="color: ${categoryColor};">🎯 Prediction Results</h3>
                            <p style="font-size: 1.2em; margin: 15px 0;"><strong>Category:</strong> <span style="color: ${categoryColor}; font-weight: bold; font-size: 1.3em;">${data.category}</span></p>
                            <p><strong>AI Confidence:</strong> <span style="color: ${categoryColor}; font-weight: bold;">${(data.confidence * 100).toFixed(1)}%</span></p>
                            <p><strong>BMI:</strong> <span style="font-weight: bold;">${data.bmi.toFixed(1)}</span> (${data.bmi_category})</p>
                            <div style="margin-top: 15px; padding: 15px; background: ${categoryColor}15; border-radius: 10px;">
                                <p style="margin: 0; font-size: 0.95em; color: #555;">
                                    ${getCategoryDescription(data.category)}
                                </p>
                            </div>
                        </div>
                        <div class="result-card" style="border-left-color: #e67e22;">
                            <h3 style="color: #e67e22;">🔍 Why You're Classified as "${data.category}"</h3>
                            <p style="margin-bottom: 15px; color: #666;">Based on your health metrics, here are the key factors:</p>
                            ${data.reasons.map((r, i) => `
                                <div style="margin: 12px 0; padding: 10px; background: #fef5e7; border-left: 3px solid #e67e22; border-radius: 5px;">
                                    <p style="margin: 0;"><strong>${i + 1}.</strong> ${r}</p>
                                </div>
                            `).join('')}
                        </div>
                        <div class="result-card" style="border-left-color: #3498db;">
                            <h3 style="color: #3498db;">📈 Top 5 Contributing Factors</h3>
                            <p style="margin-bottom: 15px; color: #666;">These factors had the biggest impact on your prediction:</p>
                            ${data.top_features.map((f, i) => {
                                const percentage = (f.importance * 100).toFixed(1);
                                const barWidth = Math.min(percentage * 2, 100);
                                return `
                                    <div style="margin: 15px 0;">
                                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                            <strong>${i + 1}. ${f.feature}:</strong>
                                            <span style="color: #3498db; font-weight: bold;">${f.value}</span>
                                        </div>
                                        <div style="background: #ecf0f1; border-radius: 10px; overflow: hidden; height: 8px;">
                                            <div style="background: linear-gradient(90deg, #3498db, #2980b9); height: 100%; width: ${barWidth}%;"></div>
                                        </div>
                                        <p style="font-size: 0.85em; color: #7f8c8d; margin-top: 3px;">Impact: ${percentage}%</p>
                                    </div>
                                `;
                            }).join('')}
                        </div>
                        <div class="result-card" style="border-left-color: #9b59b6;">
                            <h3 style="color: #9b59b6;">💡 Personalized Recommendations</h3>
                            <p style="margin-bottom: 15px; color: #666;">Follow these evidence-based recommendations to reach your health goals:</p>
                            ${data.suggestions.map((s, i) => `
                                <div style="margin: 12px 0; padding: 12px; background: #f8f4fc; border-left: 3px solid #9b59b6; border-radius: 5px;">
                                    <p style="margin: 0;"><span style="color: #9b59b6; font-weight: bold;">✓</span> ${s}</p>
                                </div>
                            `).join('')}
                        </div>
                        <div class="result-card" style="border-left-color: #16a085;">
                            <h3 style="color: #16a085;">📅 Your 8-Week Transformation Plan</h3>
                            <p style="margin-bottom: 20px; color: #666;">Follow this progressive plan for sustainable results:</p>
                            ${data.action_plan.map((a, i) => {
                                const weekMatch = a.match(/Week (\d+)-(\d+)/);
                                const isWeekPlan = weekMatch !== null;
                                return `
                                    <div style="margin: 15px 0; padding: 15px; background: ${isWeekPlan ? '#e8f8f5' : '#fff3e0'}; border-left: 4px solid ${isWeekPlan ? '#16a085' : '#f39c12'}; border-radius: 8px;">
                                        <p style="margin: 0; font-weight: ${isWeekPlan ? 'bold' : 'normal'};">${isWeekPlan ? '📆' : '🔄'} ${a}</p>
                                    </div>
                                `;
                            }).join('')}
                            <div style="margin-top: 20px; padding: 15px; background: #d5f4e6; border-radius: 10px; border: 2px dashed #16a085;">
                                <p style="margin: 0; text-align: center; font-weight: bold; color: #16a085;">
                                    💪 Remember: Consistency is key! Small daily actions lead to big results.
                                </p>
                            </div>
                        </div>
                        <div class="result-card" style="border-left-color: #e74c3c; background: linear-gradient(135deg, #fff 0%, #fee 100%);">
                            <h3 style="color: #e74c3c;">⚕️ Important Health Notice</h3>
                            <p style="margin: 10px 0; line-height: 1.8;">
                                <strong>Disclaimer:</strong> This analysis is for informational purposes only and should not replace professional medical advice. 
                                ${getCategoryWarning(data.category, data.bmi)}
                            </p>
                        </div>
                    </div>
                `;
                
                // Helper function for category descriptions
                function getCategoryDescription(category) {
                    const descriptions = {
                        'Underweight': 'Being underweight can lead to nutritional deficiencies, weakened immune system, and decreased muscle mass. It\'s important to gain weight gradually through healthy, nutrient-dense foods.',
                        'Healthy Slim': 'Congratulations! Your weight is in the healthy range. Maintain your current lifestyle habits to preserve your health and prevent future weight issues.',
                        'Overweight': 'Being overweight increases risk for heart disease, diabetes, and joint problems. Small, sustainable lifestyle changes can help you reach a healthier weight.',
                        'Obese': 'Obesity significantly increases health risks including heart disease, type 2 diabetes, sleep apnea, and certain cancers. Professional medical support can help you safely lose weight.'
                    };
                    return descriptions[category] || 'Continue monitoring your health metrics.';
                }
                
                // Helper function for category-specific warnings
                function getCategoryWarning(category, bmi) {
                    if (category === 'Underweight') {
                        return 'Please consult with a healthcare provider or registered dietitian to address potential underlying causes and develop a safe weight gain plan.';
                    } else if (category === 'Obese' || bmi >= 35) {
                        return '<strong>We strongly recommend scheduling an appointment with your healthcare provider</strong> to discuss comprehensive weight management strategies, including potential medical interventions.';
                    } else if (category === 'Overweight') {
                        return 'Consider consulting with a healthcare provider or certified nutritionist to develop a personalized weight loss plan that\'s safe and effective for you.';
                    } else {
                        return 'Continue with regular health check-ups to maintain your healthy status.';
                    }
                }

                result.classList.add('show');
                result.scrollIntoView({ behavior: 'smooth' });
            } catch (error) {
                loading.classList.remove('show');
                alert('Error: ' + error.message);
            }
        });
    </script>
</body>
</html>
"""

# Train model function
def train_model():
    global rf_multi, scaler, label_encoders, target_encoder, feature_columns, dataset_df

    print("📊 Loading dataset...")
    dataset_df = pd.read_csv('augmented_obesity_lifestyle_dataset (1).csv')

    categorical_columns = ['Gender', 'PhysicalActivity', 'FrequentConsumptionHighCalorieFood', 'FrequentVegetableConsumption']

    for col in categorical_columns:
        le = LabelEncoder()
        dataset_df[col] = le.fit_transform(dataset_df[col])
        label_encoders[col] = le

    target_encoder = LabelEncoder()
    dataset_df['Category_encoded'] = target_encoder.fit_transform(dataset_df['Category'])

    feature_columns = [
        'Height_m', 'Weight_kg', 'Age', 'Gender', 'PhysicalActivity',
        'FrequentConsumptionHighCalorieFood', 'FrequentVegetableConsumption',
        'BMI', 'Water_Intake_L', 'Sleep_Hours', 'Sleep_Quality_Score',
        'Screen_Time_Hours', 'Steps_Per_Day', 'Protein_Intake_g', 'Stress_Level_Score'
    ]

    X = dataset_df[feature_columns]
    y = dataset_df['Category_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print("🤖 Training model...")
    rf_multi = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_multi.fit(X_train_scaled, y_train)

    y_pred = rf_multi.predict(scaler.transform(X_test))
    accuracy = accuracy_score(y_test, y_pred)

    print(f"✅ Model trained! Accuracy: {accuracy:.4f}")

def analyze_reasons(category, user_data, top_features):
    """Analyze and provide detailed reasons for the slimness category"""
    reasons = []
    bmi = user_data['bmi']

    if category == 'Underweight':
        reasons.append(f"Your BMI is {bmi:.1f}, which falls in the underweight range (BMI < 18.5)")
        
        if user_data['protein'] < 60:
            reasons.append(f"Low protein intake ({user_data['protein']}g/day) - recommended minimum is 60g for your profile")
        
        if user_data['steps'] < 5000:
            reasons.append(f"Low physical activity ({user_data['steps']:,} steps) may indicate muscle loss concerns")
        
        if user_data['sleep'] < 7:
            reasons.append(f"Insufficient sleep ({user_data['sleep']} hours) can affect metabolism and appetite")
            
        if user_data['water'] < 2:
            reasons.append(f"Low water intake ({user_data['water']}L) may affect nutrient absorption")

    elif category == 'Healthy Slim':
        reasons.append(f"Your BMI is {bmi:.1f}, which is in the healthy weight range (18.5-24.9)")
        
        if user_data['steps'] >= 8000:
            reasons.append(f"Excellent activity level with {user_data['steps']:,} daily steps")
        
        if user_data['protein'] >= 60:
            reasons.append(f"Good protein intake ({user_data['protein']}g/day) supporting lean muscle mass")
        
        if user_data['vegetables'] == 'yes':
            reasons.append("Regular vegetable consumption provides essential micronutrients")
            
        if user_data['sleep'] >= 7:
            reasons.append(f"Adequate sleep ({user_data['sleep']} hours) supporting metabolic health")

    elif category == 'Overweight':
        reasons.append(f"Your BMI is {bmi:.1f}, which falls in the overweight range (25-29.9)")
        
        if user_data['steps'] < 7000:
            reasons.append(f"Below-average activity level ({user_data['steps']:,} steps) - target is 10,000+ steps")
        
        if user_data['highCalorie'] == 'yes':
            reasons.append("Frequent high-calorie food consumption contributing to caloric surplus")
        
        if user_data['screenTime'] > 6:
            reasons.append(f"High screen time ({user_data['screenTime']} hours) associated with sedentary lifestyle")
        
        if user_data['activity'] in ['Sedentary', 'Light']:
            reasons.append(f"Low physical activity level ({user_data['activity']}) limits calorie expenditure")

    elif category == 'Obese':
        reasons.append(f"Your BMI is {bmi:.1f}, which falls in the obese range (BMI ≥ 30)")
        
        if user_data['steps'] < 5000:
            reasons.append(f"Very low activity level ({user_data['steps']:,} steps) - significantly below recommended")
        
        if user_data['highCalorie'] == 'yes':
            reasons.append("Frequent consumption of high-calorie foods creating substantial caloric surplus")
        
        if user_data['vegetables'] == 'no':
            reasons.append("Low vegetable consumption - missing fiber and nutrients that aid weight management")
        
        if user_data['sleep'] < 7 or user_data['sleep'] > 9:
            reasons.append(f"Suboptimal sleep duration ({user_data['sleep']} hours) can disrupt hormones regulating appetite")
        
        if user_data['stress'] >= 7:
            reasons.append(f"High stress levels ({user_data['stress']}/10) can trigger emotional eating and cortisol-related weight gain")

    return reasons

def get_suggestions(category, user_data):
    """Generate personalized suggestions and action plan"""
    suggestions = []
    action_plan = []

    if category == 'Underweight':
        suggestions.append("Increase caloric intake by 300-500 calories daily through nutrient-dense foods")
        suggestions.append(f"Boost protein intake from {user_data['protein']}g to 80-100g daily (eggs, lean meats, legumes)")
        suggestions.append("Add healthy fats: nuts, avocados, olive oil (calorie-dense and nutritious)")
        suggestions.append("Eat 5-6 smaller meals throughout the day to increase total intake")
        
        if user_data['steps'] < 5000:
            suggestions.append("Moderate exercise with focus on strength training to build muscle mass")
        
        action_plan.extend([
            "Week 1-2: Add protein shake between meals, increase portions by 20%",
            "Week 3-4: Start strength training 3x/week, track calorie intake daily",
            "Week 5-6: Add healthy snacks (nuts, dried fruits), continue building muscle",
            "Week 7-8: Reassess progress, adjust caloric intake based on weight gain"
        ])

    elif category == 'Healthy Slim':
        suggestions.append("Maintain current healthy habits - you're doing great!")
        suggestions.append(f"Continue balanced nutrition with adequate protein ({user_data['protein']}g)")
        suggestions.append("Stay hydrated with 2-3L water daily")
        suggestions.append("Keep up regular physical activity and sleep routine")
        
        if user_data['stress'] >= 6:
            suggestions.append("Practice stress management: meditation, yoga, or deep breathing exercises")
        
        action_plan.extend([
            "Week 1-2: Monitor weight weekly, maintain current routines",
            "Week 3-4: Add variety to exercise routine to prevent plateaus",
            "Week 5-6: Try new healthy recipes to maintain dietary enjoyment",
            "Week 7-8: Set new fitness goals (flexibility, endurance, or strength)"
        ])

    elif category == 'Overweight':
        suggestions.append(f"Create calorie deficit: reduce daily intake by 300-500 calories for gradual weight loss")
        suggestions.append(f"Increase daily steps from {user_data['steps']:,} to 10,000+ gradually")
        suggestions.append("Replace high-calorie processed foods with whole foods (fruits, vegetables, lean proteins)")
        
        if user_data['water'] < 2.5:
            suggestions.append(f"Increase water intake from {user_data['water']}L to 2.5-3L daily to boost metabolism")
        
        if user_data['vegetables'] == 'no':
            suggestions.append("Add vegetables to every meal - they're low-calorie and high in fiber")
        
        if user_data['screenTime'] > 5:
            suggestions.append(f"Reduce screen time from {user_data['screenTime']} to under 4 hours, use saved time for activity")
        
        action_plan.extend([
            "Week 1-2: Track all food intake, add 2,000 extra steps daily, drink water before meals",
            "Week 3-4: Increase steps to 8,000+, swap one processed meal for whole foods daily",
            "Week 5-6: Start 30-min cardio 3x/week, aim for 10,000 steps, meal prep healthy lunches",
            "Week 7-8: Add strength training 2x/week, continue cardio, target 1-2 lbs weight loss/week"
        ])

    elif category == 'Obese':
        suggestions.append("Create sustainable calorie deficit of 500-750 calories daily for 1-2 lbs/week weight loss")
        suggestions.append(f"Start movement program: increase steps from {user_data['steps']:,} to 5,000, then 7,500, then 10,000")
        suggestions.append("Eliminate high-calorie processed foods and sugary beverages completely")
        suggestions.append("Focus on whole foods: lean proteins, vegetables, fruits, whole grains, healthy fats")
        
        if user_data['protein'] < 80:
            suggestions.append(f"Increase protein to 80-100g daily to preserve muscle during weight loss")
        
        if user_data['vegetables'] == 'no':
            suggestions.append("Make vegetables half of every meal - fills you up with minimal calories")
        
        if user_data['sleep'] < 7:
            suggestions.append(f"Prioritize sleep: increase from {user_data['sleep']} to 7-9 hours to regulate hunger hormones")
        
        if user_data['stress'] >= 7:
            suggestions.append("Address stress through counseling, meditation, or support groups to prevent emotional eating")
        
        suggestions.append("Consider consulting healthcare provider for comprehensive weight management plan")
        
        action_plan.extend([
            "Week 1-2: Start food journal, walk 10 min after each meal, remove junk food from home",
            "Week 3-4: Increase walking to 20 min 3x/day, meal prep on Sundays, drink 8 glasses water",
            "Week 5-6: Join activity class or gym, increase to 5,000+ steps, focus on protein at meals",
            "Week 7-8: Work toward 7,500 steps, add light strength training, track weekly progress photos"
        ])

    # Add sleep recommendations if needed
    if user_data['sleep'] < 7:
        suggestions.append(f"Improve sleep from {user_data['sleep']} to 7-9 hours: set consistent bedtime, limit screens 1hr before bed")
        action_plan.append("Ongoing: Establish bedtime routine, keep bedroom cool and dark, avoid caffeine after 2pm")
    
    # Add water recommendations if needed
    if user_data['water'] < 2:
        suggestions.append(f"Double water intake from {user_data['water']}L to 2.5-3L: carry water bottle, drink before meals")
        action_plan.append("Daily: Set hourly water reminders, flavor with lemon/cucumber if plain water is boring")

    return suggestions, action_plan

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        bmi = data['weight'] / (data['height'] ** 2)

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

        user_input_scaled = scaler.transform([user_input])
        prediction = rf_multi.predict(user_input_scaled)[0]
        probabilities = rf_multi.predict_proba(user_input_scaled)[0]

        category = target_encoder.inverse_transform([prediction])[0]
        confidence = probabilities[prediction]

        if bmi < 18.5:
            bmi_category = "Underweight"
        elif 18.5 <= bmi < 25:
            bmi_category = "Normal weight"
        elif 25 <= bmi < 30:
            bmi_category = "Overweight"
        else:
            bmi_category = "Obese"

        importances = rf_multi.feature_importances_
        top_indices = np.argsort(importances)[-5:][::-1]
        top_features = []
        for idx in top_indices:
            feature_name = feature_columns[idx]
            feature_value = user_input[idx]
            
            # Make feature names more readable
            readable_names = {
                'Height_m': 'Height',
                'Weight_kg': 'Weight',
                'Age': 'Age',
                'Gender': 'Gender',
                'PhysicalActivity': 'Activity Level',
                'FrequentConsumptionHighCalorieFood': 'High-Calorie Food Intake',
                'FrequentVegetableConsumption': 'Vegetable Consumption',
                'BMI': 'Body Mass Index',
                'Water_Intake_L': 'Water Intake',
                'Sleep_Hours': 'Sleep Duration',
                'Sleep_Quality_Score': 'Sleep Quality',
                'Screen_Time_Hours': 'Screen Time',
                'Steps_Per_Day': 'Daily Steps',
                'Protein_Intake_g': 'Protein Intake',
                'Stress_Level_Score': 'Stress Level'
            }
            
            top_features.append({
                'feature': readable_names.get(feature_name, feature_name),
                'value': round(feature_value, 2) if isinstance(feature_value, float) else feature_value,
                'importance': importances[idx]
            })

        user_data_with_bmi = data.copy()
        user_data_with_bmi['bmi'] = bmi
        reasons = analyze_reasons(category, user_data_with_bmi, top_features)
        suggestions, action_plan = get_suggestions(category, user_data_with_bmi)

        analysis_results = {
            'category': category,
            'confidence': float(confidence),
            'bmi': float(bmi),
            'bmi_category': bmi_category,
            'top_features': top_features,
            'reasons': reasons,
            'suggestions': suggestions,
            'action_plan': action_plan
        }

        patient_record = data.copy()
        patient_record['analysis_results'] = analysis_results
        patient_record['timestamp'] = datetime.datetime.now().isoformat()
        save_patient_record(patient_record)

        return jsonify(analysis_results)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/get_visualizations')
def get_visualizations():
    global dataset_df

    chart_base64 = generate_pie_charts()

    # Read original dataset for gender stats
    df_original = pd.read_csv('augmented_obesity_lifestyle_dataset (1).csv')
    
    stats = {
        'total_records': len(df_original),
        'avg_age': f"{df_original['Age'].mean():.1f}",
        'avg_bmi': f"{df_original['BMI'].mean():.1f}",
        'male_percent': f"{(df_original['Gender'].value_counts().get('Male', 0) / len(df_original) * 100):.1f}",
        'female_percent': f"{(df_original['Gender'].value_counts().get('Female', 0) / len(df_original) * 100):.1f}"
    }

    return jsonify({
        'chart': chart_base64,
        'stats': stats
    })

@app.route('/get_dataset')
def get_dataset():
    df_original = pd.read_csv('augmented_obesity_lifestyle_dataset (1).csv')

    columns = df_original.columns.tolist()
    rows = df_original.values.tolist()

    return jsonify({
        'columns': columns,
        'rows': rows,
        'total_rows': len(rows)
    })

# Main execution
if __name__ == '__main__':
    print("🚀 Starting Health Analysis Application...")
    print("="*60)

    init_data_storage()
    train_model()

    # Start ngrok tunnel - FIXED VERSION
    port = 5000
    public_url = ngrok.connect(port).public_url  # FIXED: Added .public_url

    print("\n" + "="*60)
    print("✅ APPLICATION IS RUNNING!")
    print("="*60)
    print(f"🌐 Public URL: {public_url}")
    print(f"📍 Local URL: http://127.0.0.1:{port}")
    print("="*60)
    print("\n👉 Click the public URL to access your app!")
    print("📊 Features: Prediction | Data Visualization | Dataset View")
    print("Press CTRL+C to stop\n")

    app.run(port=port)
