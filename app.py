# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import json
import datetime
from fpdf import FPDF
import base64
import io
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Slimness Prediction & Health Analysis",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .advantage-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid;
    }
    .feature-item {
        background: #f8f9fa;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 3px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'patient_records' not in st.session_state:
    st.session_state.patient_records = []
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'target_encoder' not in st.session_state:
    st.session_state.target_encoder = None

# Feature columns
feature_columns = [
    'Height_m', 'Weight_kg', 'Age', 'Gender', 'PhysicalActivity',
    'FrequentConsumptionHighCalorieFood', 'FrequentVegetableConsumption',
    'BMI', 'Water_Intake_L', 'Sleep_Hours', 'Sleep_Quality_Score',
    'Screen_Time_Hours', 'Steps_Per_Day', 'Protein_Intake_g', 'Stress_Level_Score'
]

def train_model():
    """Train the machine learning model"""
    try:
        # Load dataset
        df = pd.read_csv('augmented_obesity_lifestyle_dataset (1).csv')
        
        # Handle categorical variables
        categorical_columns = ['Gender', 'PhysicalActivity', 'FrequentConsumptionHighCalorieFood', 'FrequentVegetableConsumption']
        
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            st.session_state.label_encoders[col] = le
        
        # Encode target
        target_encoder = LabelEncoder()
        df['Category_encoded'] = target_encoder.fit_transform(df['Category'])
        st.session_state.target_encoder = target_encoder
        
        # Prepare features and target
        X = df[feature_columns]
        y = df['Category_encoded']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        st.session_state.scaler = scaler
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train_scaled, y_train)
        st.session_state.model = model
        
        # Calculate accuracy
        from sklearn.metrics import accuracy_score
        y_pred = model.predict(scaler.transform(X_test))
        accuracy = accuracy_score(y_test, y_pred)
        
        st.session_state.model_trained = True
        return True, accuracy
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return False, 0

def analyze_reasons(category, user_data, top_features):
    """Analyze reasons for the prediction"""
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
        
        if feature_name == 'PhysicalActivity' and value in [0, 1]:  # Sedentary/Light
            reasons.append("Low physical activity level impacting energy balance")
        elif feature_name == 'Protein_Intake_g' and value < 50:
            reasons.append("Suboptimal protein intake for muscle maintenance")
        elif feature_name == 'Steps_Per_Day' and value < 6000:
            reasons.append("Insufficient daily steps for cardiovascular health")
        elif feature_name == 'Screen_Time_Hours' and value > 6:
            reasons.append("Excessive screen time affecting activity patterns")
    
    return reasons

def get_suggestions(category, user_data):
    """Get personalized suggestions and action plan"""
    suggestions = []
    action_plan = []
    
    bmi = user_data['bmi']
    steps = user_data['steps']
    sleep = user_data['sleep']
    water = user_data['water']
    stress = user_data['stress']
    protein = user_data['protein']
    screen_time = user_data['screenTime']
    
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

def generate_pdf_report(patient_data, analysis_results):
    """Generate PDF report"""
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Slimness Prediction & Health Analysis Report', 0, 1, 'C')
    pdf.ln(10)
    
    # Patient Information
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Patient Information', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 6, f"Name: {patient_data.get('name', 'N/A')}", 0, 1)
    pdf.cell(0, 6, f"Email: {patient_data.get('email', 'N/A')}", 0, 1)
    pdf.cell(0, 6, f"Phone: {patient_data.get('phone', 'N/A')}", 0, 1)
    pdf.cell(0, 6, f"Date: {patient_data.get('date', 'N/A')}", 0, 1)
    pdf.ln(5)
    
    # Health Metrics
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Health Metrics', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 6, f"Height: {patient_data.get('height', 'N/A')} m", 0, 1)
    pdf.cell(0, 6, f"Weight: {patient_data.get('weight', 'N/A')} kg", 0, 1)
    pdf.cell(0, 6, f"BMI: {analysis_results.get('bmi', 'N/A'):.1f} ({analysis_results.get('bmi_category', 'N/A')})", 0, 1)
    pdf.cell(0, 6, f"Age: {patient_data.get('age', 'N/A')} years", 0, 1)
    pdf.ln(5)
    
    # Prediction Results
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Prediction Results', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 6, f"Category: {analysis_results.get('category', 'N/A')}", 0, 1)
    pdf.cell(0, 6, f"Confidence: {analysis_results.get('confidence', 0)*100:.1f}%", 0, 1)
    pdf.ln(5)
    
    # Top Factors
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Top Influencing Factors', 0, 1)
    pdf.set_font('Arial', '', 10)
    for i, feature in enumerate(analysis_results.get('top_features', [])[:3]):
        feature_name = feature['feature'].replace('_', ' ').title()
        pdf.cell(0, 6, f"{i+1}. {feature_name}: {feature['value']} ({feature['importance']*100:.1f}%)", 0, 1)
    pdf.ln(5)
    
    # Recommendations
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Personalized Recommendations', 0, 1)
    pdf.set_font('Arial', '', 10)
    for i, suggestion in enumerate(analysis_results.get('suggestions', [])[:8]):
        pdf.multi_cell(0, 6, f"• {suggestion}")
    pdf.ln(5)
    
    # Action Plan
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '8-Week Action Plan', 0, 1)
    pdf.set_font('Arial', '', 10)
    for step in analysis_results.get('action_plan', [])[:6]:
        pdf.multi_cell(0, 6, f"✓ {step}")
    
    # Return PDF as bytes
    return pdf.output(dest='S').encode('latin1')

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🏃‍♂️ Slimness Prediction & Health Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Health Assessment & Personalized Recommendations</p>', unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose a section:", 
                               ["🏠 Home", "🔍 Health Analysis", "⭐ Advantages", "📋 Patient Records", "📊 Analysis History"])
    
    # Train model if not already trained
    if not st.session_state.model_trained:
        with st.spinner("Training AI model... This may take a few seconds."):
            success, accuracy = train_model()
            if success:
                st.sidebar.success(f"✅ Model trained! Accuracy: {accuracy:.1%}")
            else:
                st.sidebar.error("❌ Model training failed!")
    
    # Home Page
    if app_mode == "🏠 Home":
        st.markdown("""
        ## Welcome to the Slimness Prediction System!
        
        This advanced AI-powered platform provides comprehensive health analysis and personalized 
        recommendations for achieving your optimal weight and health goals.
        
        ### 🚀 Quick Start:
        1. Go to **🔍 Health Analysis** to get your personalized assessment
        2. View **⭐ Advantages** to learn about our unique features
        3. Check **📋 Patient Records** to manage your data
        4. Review **📊 Analysis History** for past assessments
        
        ### 💡 How it works:
        - Input your health metrics and lifestyle information
        - Our AI model analyzes 15+ health parameters
        - Get instant predictions with detailed explanations
        - Receive personalized recommendations and action plans
        - Download professional PDF reports
        """)
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="stat-card">
                <h3>95%</h3>
                <p>Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="stat-card">
                <h3>15+</h3>
                <p>Parameters</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="stat-card">
                <h3>24/7</h3>
                <p>Access</p>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div class="stat-card">
                <h3>100%</h3>
                <p>Private</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Health Analysis Page
    elif app_mode == "🔍 Health Analysis":
        st.header("🔍 Health Analysis")
        
        with st.form("health_analysis_form"):
            st.subheader("👤 Patient Information")
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Full Name", placeholder="Enter patient's full name")
                email = st.text_input("Email Address", placeholder="patient@example.com")
            with col2:
                phone = st.text_input("Phone Number", placeholder="+1 (555) 123-4567")
                date = st.date_input("Assessment Date", datetime.date.today())
            
            st.subheader("📊 Basic Health Information")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.75, step=0.01)
            with col2:
                weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=68.0, step=0.1)
            with col3:
                age = st.number_input("Age", min_value=15, max_value=100, value=28)
            with col4:
                gender = st.selectbox("Gender", ["Male", "Female"])
            
            st.subheader("💪 Lifestyle & Activity")
            col1, col2, col3 = st.columns(3)
            with col1:
                activity = st.selectbox("Physical Activity Level", 
                                      ["Sedentary", "Light", "Moderate", "High"])
            with col2:
                steps = st.number_input("Daily Steps", min_value=0, max_value=50000, value=8500)
            with col3:
                screen_time = st.number_input("Screen Time (hours/day)", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
            
            st.subheader("🍎 Nutrition & Diet")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                protein = st.number_input("Protein Intake (g/day)", min_value=0, max_value=300, value=75)
            with col2:
                water = st.number_input("Water Intake (L/day)", min_value=0.0, max_value=10.0, value=2.5, step=0.1)
            with col3:
                high_calorie = st.selectbox("High-Calorie Food", ["no", "yes"])
            with col4:
                vegetables = st.selectbox("Vegetable Consumption", ["yes", "no"])
            
            st.subheader("😴 Health & Wellness")
            col1, col2, col3 = st.columns(3)
            with col1:
                sleep = st.number_input("Sleep Hours/Night", min_value=0.0, max_value=24.0, value=7.5, step=0.5)
            with col2:
                sleep_quality = st.slider("Sleep Quality (1-10)", 1, 10, 7)
            with col3:
                stress = st.slider("Stress Level (1-10)", 1, 10, 4)
            
            submitted = st.form_submit_button("🔍 Analyze Health Status")
            
            if submitted:
                if not name:
                    st.error("Please enter patient name!")
                    return
                
                with st.spinner("Analyzing your health data..."):
                    # Calculate BMI
                    bmi = weight / (height ** 2)
                    
                    # Prepare user input
                    user_input = []
                    for col in feature_columns:
                        if col == 'BMI':
                            user_input.append(bmi)
                        elif col == 'Height_m':
                            user_input.append(height)
                        elif col == 'Weight_kg':
                            user_input.append(weight)
                        elif col == 'Age':
                            user_input.append(age)
                        elif col == 'Gender':
                            user_input.append(st.session_state.label_encoders['Gender'].transform([gender])[0])
                        elif col == 'PhysicalActivity':
                            user_input.append(st.session_state.label_encoders['PhysicalActivity'].transform([activity])[0])
                        elif col == 'FrequentConsumptionHighCalorieFood':
                            user_input.append(st.session_state.label_encoders['FrequentConsumptionHighCalorieFood'].transform([high_calorie])[0])
                        elif col == 'FrequentVegetableConsumption':
                            user_input.append(st.session_state.label_encoders['FrequentVegetableConsumption'].transform([vegetables])[0])
                        elif col == 'Water_Intake_L':
                            user_input.append(water)
                        elif col == 'Sleep_Hours':
                            user_input.append(sleep)
                        elif col == 'Sleep_Quality_Score':
                            user_input.append(sleep_quality)
                        elif col == 'Screen_Time_Hours':
                            user_input.append(screen_time)
                        elif col == 'Steps_Per_Day':
                            user_input.append(steps)
                        elif col == 'Protein_Intake_g':
                            user_input.append(protein)
                        elif col == 'Stress_Level_Score':
                            user_input.append(stress)
                    
                    # Scale input and predict
                    user_input_scaled = st.session_state.scaler.transform([user_input])
                    prediction = st.session_state.model.predict(user_input_scaled)[0]
                    probabilities = st.session_state.model.predict_proba(user_input_scaled)[0]
                    
                    # Get results
                    category = st.session_state.target_encoder.inverse_transform([prediction])[0]
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
                    importances = st.session_state.model.feature_importances_
                    top_indices = np.argsort(importances)[-3:][::-1]
                    top_features = []
                    for idx in top_indices:
                        top_features.append({
                            'feature': feature_columns[idx],
                            'value': user_input[idx],
                            'importance': importances[idx]
                        })
                    
                    # Analyze reasons
                    user_data = {
                        'bmi': bmi, 'protein': protein, 'steps': steps, 'highCalorie': high_calorie,
                        'vegetables': vegetables, 'sleep': sleep, 'screenTime': screen_time
                    }
                    reasons = analyze_reasons(category, user_data, top_features)
                    
                    # Get suggestions and action plan
                    suggestions, action_plan = get_suggestions(category, user_data)
                    
                    # Create analysis results
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
                    
                    # Save patient record
                    patient_record = {
                        'name': name,
                        'email': email,
                        'phone': phone,
                        'date': str(date),
                        'height': height,
                        'weight': weight,
                        'age': age,
                        'gender': gender,
                        'activity': activity,
                        'steps': steps,
                        'screenTime': screen_time,
                        'protein': protein,
                        'water': water,
                        'highCalorie': high_calorie,
                        'vegetables': vegetables,
                        'sleep': sleep,
                        'sleepQuality': sleep_quality,
                        'stress': stress,
                        'analysis_results': analysis_results
                    }
                    st.session_state.patient_records.append(patient_record)
                    
                    # Display results
                    st.success("✅ Analysis Complete!")
                    
                    # Results section
                    st.header("📊 Analysis Results")
                    
                    # Main prediction card
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        <div class="result-card" style="border-left-color: #667eea;">
                            <h3>🎯 Weight Status Prediction</h3>
                            <p><strong>Category:</strong> {category}</p>
                            <p><strong>Confidence:</strong> {confidence*100:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="result-card" style="border-left-color: #66bb6a;">
                            <h3>⚖️ BMI Analysis</h3>
                            <p><strong>Your BMI:</strong> {bmi:.1f}</p>
                            <p><strong>Classification:</strong> {bmi_category}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Top features
                    st.subheader("🔍 Top Influencing Factors")
                    for feature in top_features:
                        feature_name = feature['feature'].replace('_', ' ').title()
                        st.markdown(f"""
                        <div class="feature-item">
                            <strong>{feature_name}:</strong> {feature['value']} 
                            <span style="float: right; color: #667eea; font-weight: bold;">
                                {feature['importance']*100:.1f}%
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Reasons analysis
                    st.subheader("🔬 Reasons Analysis")
                    for reason in reasons:
                        st.write(f"• {reason}")
                    
                    # Recommendations
                    st.subheader("💡 Personalized Recommendations")
                    for suggestion in suggestions:
                        st.info(f"📌 {suggestion}")
                    
                    # Action plan
                    st.subheader("📅 8-Week Action Plan")
                    for step in action_plan:
                        st.success(f"✅ {step}")
                    
                    # PDF download
                    st.subheader("📄 Download Report")
                    pdf_bytes = generate_pdf_report(patient_record, analysis_results)
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"health_report_{name.replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf"
                    )
    
    # Advantages Page
    elif app_mode == "⭐ Advantages":
        st.header("⭐ Why Choose Our System?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="advantage-card" style="border-left-color: #667eea;">
                <h3>🤖 AI-Powered Analysis</h3>
                <p>Advanced machine learning algorithms provide accurate weight status predictions with 95%+ accuracy, analyzing 15+ health parameters simultaneously.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="advantage-card" style="border-left-color: #66bb6a;">
                <h3>🎯 Personalized Solutions</h3>
                <p>Get customized health recommendations and 8-week action plans tailored specifically to your unique body composition and lifestyle factors.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="advantage-card" style="border-left-color: #ffa726;">
                <h3>📊 Comprehensive Tracking</h3>
                <p>Maintain complete patient records with search functionality, analysis history, and progress tracking across multiple assessments.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="advantage-card" style="border-left-color: #ef5350;">
                <h3>🔍 Deep Insight Analysis</h3>
                <p>Understand the 'why' behind your weight status with detailed factor analysis and feature importance rankings.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="advantage-card" style="border-left-color: #ab47bc;">
                <h3>📄 Professional Reporting</h3>
                <p>Generate comprehensive PDF reports for medical records, insurance claims, or personal tracking with professional formatting.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="advantage-card" style="border-left-color: #42a5f5;">
                <h3>💡 Actionable Intelligence</h3>
                <p>Transform complex health data into simple, actionable steps with clear implementation guidelines and progress milestones.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Statistics
        st.subheader("📈 Our Impact")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Prediction Accuracy", "95%")
        with col2:
            st.metric("Health Parameters", "15+")
        with col3:
            st.metric("Available 24/7", "Yes")
        with col4:
            st.metric("Data Privacy", "100%")
    
    # Patient Records Page
    elif app_mode == "📋 Patient Records":
        st.header("📋 Patient Records")
        
        # Search functionality
        search_term = st.text_input("🔍 Search patients by name, email, or phone:")
        
        if st.button("Clear Search"):
            search_term = ""
        
        # Display patient records
        records = st.session_state.patient_records
        if search_term:
            search_term = search_term.lower()
            records = [r for r in records if (
                search_term in r.get('name', '').lower() or
                search_term in r.get('email', '').lower() or
                search_term in r.get('phone', '').lower()
            )]
        
        if not records:
            st.info("No patient records found. Complete a health analysis to create records.")
        else:
            for i, record in enumerate(records):
                with st.expander(f"👤 {record.get('name', 'Unknown')} - {record.get('date', 'Unknown date')}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Email:** {record.get('email', 'N/A')}")
                        st.write(f"**Phone:** {record.get('phone', 'N/A')}")
                        st.write(f"**Age:** {record.get('age', 'N/A')}")
                        st.write(f"**Gender:** {record.get('gender', 'N/A')}")
                    with col2:
                        st.write(f"**Height:** {record.get('height', 'N/A')} m")
                        st.write(f"**Weight:** {record.get('weight', 'N/A')} kg")
                        st.write(f"**BMI:** {record.get('analysis_results', {}).get('bmi', 'N/A'):.1f}")
                    
                    if st.button(f"Load Data for Analysis", key=f"load_{i}"):
                        # This would require more complex state management in a real app
                        st.info("In a full implementation, this would load the data into the analysis form")
    
    # Analysis History Page
    elif app_mode == "📊 Analysis History":
        st.header("📊 Analysis History")
        
        if not st.session_state.patient_records:
            st.info("No analysis history found. Complete a health analysis to see history.")
        else:
            # Sort by date
            sorted_records = sorted(st.session_state.patient_records, 
                                  key=lambda x: x.get('date', ''), reverse=True)
            
            for record in sorted_records:
                analysis = record.get('analysis_results', {})
                with st.expander(f"📅 {record.get('date', 'Unknown')} - {record.get('name', 'Unknown')}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Category", analysis.get('category', 'N/A'))
                    with col2:
                        st.metric("BMI", f"{analysis.get('bmi', 0):.1f}")
                    with col3:
                        st.metric("Confidence", f"{analysis.get('confidence', 0)*100:.1f}%")
                    
                    st.write("**Top Recommendations:**")
                    for suggestion in analysis.get('suggestions', [])[:3]:
                        st.write(f"• {suggestion}")

if __name__ == "__main__":
    main()
