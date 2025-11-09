import pandas as pd
import numpy as np
import streamlit as st
import google.generativeai as genai
import speech_recognition as sr
from gtts import gTTS
import uuid
import os
import ast
import random
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# ---------------------------
# 🎨 Modern UI Styling
# ---------------------------
st.markdown('''
    <style>
    body, .stApp {
        # background: #23255a;
        color: #fff;
    }
    .main-title {
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0.2em;
        letter-spacing: 1px;
    }
    .subtitle {
        font-size: 1.2em;
        margin-bottom: 1.5em;
        color: #b3b3ff;
    }
    .section {
        background: rgba(255,255,255,0.05);
        border-radius: 20px;
        padding: 2em;
        margin-bottom: 2em;
        box-shadow: 0 2px 12px rgba(0,0,0,0.1);
    }
    </style>
''', unsafe_allow_html=True)

st.markdown('<div class="main-title">Linda</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Driven Data Solutions for Future-Ready Organizations</div>', unsafe_allow_html=True)

# ---------------------------
# 🔑 Helper Functions
# ---------------------------

recognizer = sr.Recognizer()
def get_voice_input():
    with sr.Microphone() as source:
        st.info("🎤 Listening...")
        audio = recognizer.listen(source, timeout=5)
    try:
        query = recognizer.recognize_google(audio)
        st.success(f"You said: {query}")
        return query
    except Exception as e:
        st.error(f"Voice error: {e}")
        return ""

def speak_text(text):
    tts = gTTS(text=text, lang="en")
    tts.save("result.mp3")
    os.system("start result.mp3")

def ask_ai(question, df):
    genai.configure(api_key="AIzaSyADkL6_0GAaUgT9MzWoEGbTJkZfLgWtSE0")
    model = genai.GenerativeModel("models/gemini-2.0-flash")

    prompt = f"""
    You are a pandas assistant.
    The dataframe has these columns: {list(df.columns)}.
    Translate the user question into a valid single-line Pandas expression
    using 'df'. Be case-insensitive with column names.
    Only output the code, no explanation.
    Question: {question}
    """

    response = model.generate_content(prompt)
    code = response.text.strip()

    if code.startswith("```"):
        code = code.strip("`")
    code_lines = [line.strip() for line in code.splitlines() 
                  if line.strip() and not line.strip().startswith('#')]
    valid_code = ''
    for line in code_lines:
        if line.lower() in ['python', 'pandas', 'code']:
            continue
        valid_code = line
        break
    return valid_code

def fix_column_names(df, code):
    mapping = {col.lower(): col for col in df.columns}
    for low, real in mapping.items():
        code = code.replace(f"['{low}']", f"['{real}']")
        code = code.replace(f'["{low}"]', f'["{real}"]')
    return code

def train_recommender(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col].values.ravel()  # Fix here: make y 1D array

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", model)
    ])

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Train-test split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.3, random_state=42, stratify=y_enc
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_probs = pipeline.predict_proba(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "roc_auc": roc_auc_score(y_test, y_probs, multi_class="ovr")
    }

    return {"pipeline": pipeline, "le": le, "metrics": metrics, "y_test": y_test, "y_probs": y_probs}


def prepare_input(user_input, df, target_col):
    full_cols = df.drop(columns=[target_col]).columns
    # Convert None to np.nan for safe filling
    user_input_clean = {k: (np.nan if v is None else v) for k, v in user_input.items()}
    input_df = pd.DataFrame([user_input_clean])

    for col in full_cols:
        if col not in input_df.columns or pd.isna(input_df.at[0, col]):
            if df[col].dtype == "object":
                input_df[col] = df[col].mode()[0]
            else:
                input_df[col] = df[col].mean()
    return input_df[full_cols]

def prepare_input(user_input, df, target_col):
    full_cols = df.drop(columns=[target_col]).columns
    input_df = pd.DataFrame([user_input])

    for col in full_cols:
        if col not in input_df.columns:
            if df[col].dtype == "object":
                input_df[col] = df[col].mode()[0]
            else:
                input_df[col] = df[col].mean()
    return input_df[full_cols]

def collect_user_inputs(df, exclude_cols=[], dataset_name="dataset"):
    sample = {}
    for col in df.columns:
        if col in exclude_cols: 
            continue
        if df[col].dtype == "object":
            sample[col] = st.selectbox(
                f"{col}", df[col].unique(), key=f"{dataset_name}_{col}"
            )
        else:
            sample[col] = st.number_input(
                f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()),
                key=f"{dataset_name}_{col}"
            )
    return pd.DataFrame([sample])

def show_model_metrics(recomm):
    metrics = recomm["metrics"]
    st.markdown("### 📈 Model Performance")
    st.table(pd.DataFrame([metrics]))

    # ROC Curve
    fpr, tpr, _ = roc_curve(recomm["y_test"], recomm["y_probs"][:,1], pos_label=1)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC'))
    fig_roc.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash'))
    fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig_roc, use_container_width=True)

    # Feature Importance
    clf = recomm["pipeline"].named_steps["clf"]
    feat_names = recomm["pipeline"].named_steps["preprocessor"].get_feature_names_out()
    importances = clf.feature_importances_
    feat_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False).head(10)
    st.bar_chart(feat_df.set_index("feature"))

# ---------------------------
# 📂 Load Datasets
# ---------------------------
df_life = pd.read_csv("life_health_insurance_dataset - life_health_insurance_dataset.csv")
df_motor = pd.read_csv("LINDA_ Motors - motor_insurance_dataset.csv")

# Synthetic dataset
def generate_synthetic_health(n=200):
    genders = ["Male", "Female"]
    marital_statuses = ["Single", "Married", "Divorced", "Widowed"]
    occupations = ["Salaried", "Self-Employed", "Business", "Student", "Retired"]
    income_levels = ["Low", "Medium", "High"]
    education_levels = ["High School", "Graduate", "Postgraduate", "Doctorate"]
    locations = ["Urban", "Semi-Urban", "Rural"]
    health_conditions = ["None", "Diabetes", "Hypertension", "Heart Disease", "Asthma", "Cancer"]
    smoking_status = ["Yes", "No"]
    alcohol_status = ["Yes", "No"]
    claim_history = ["Yes", "No"]
    lifestyles = ["Active", "Sedentary", "Moderate"]
    risk_appetite = ["Low", "Medium", "High"]
    travel_frequency = ["Rarely", "Often", "Frequent"]
    addons = ["Maternity", "Accidental Cover", "Critical Illness", "Hospital Cash", "None"]
    recommended_products = ["Individual", "Family Floater", "Senior Citizen", "Critical Illness"]

    def generate_record(i):
        return {
            "Customer_ID": f"CUST{i:04d}",
            "Age": random.randint(18, 70),
            "Gender": random.choice(genders),
            "Marital_Status": random.choice(marital_statuses),
            "Occupation": random.choice(occupations),
            "Income_Level": random.choice(income_levels),
            "Education": random.choice(education_levels),
            "Location": random.choice(locations),
            "Family_Size": random.randint(1, 6),
            "Health_Condition": random.choice(health_conditions),
            "Smoking": random.choice(smoking_status),
            "Alcohol": random.choice(alcohol_status),
            "Claim_History": random.choice(claim_history),
            "Lifestyle": random.choice(lifestyles),
            "Risk_Appetite": random.choice(risk_appetite),
            "Travel_Frequency": random.choice(travel_frequency),
            "Coverage_Amount": random.randint(200000, 2000000),
            "Premium_Budget": random.randint(5000, 50000),
            "Tenure_Preference": random.choice([5, 10, 15, 20, 25]),
            "Addons_Interested": random.choice(addons),
            "Recommended_Product": random.choice(recommended_products)
        }

    return pd.DataFrame([generate_record(i) for i in range(1, n+1)])

df_synth = generate_synthetic_health()

# ---------------------------
# 📑 Tabs
# ---------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Home",
    "📊 Data Explorer", 
    "🏥 Life Recommendation", 
    "🚗 Motor Recommendation",
    "🧪  Health Recommendation"
])

# ---------------------------
# 📊 Tab 1: Data Explorer
# ---------------------------

with tab1:
    st.markdown("<div class='main-title'>Linda – Voice-Ready AI Assistant</div>", unsafe_allow_html=True)
    st.markdown("""
    **Linda** is a voice- and text-based AI chatbot designed for fast, intelligent support across **Web, Mobile, and WhatsApp**.<br>
    Whether your customers prefer speaking or typing, Linda delivers accurate and insightful help—24/7, in any context.
    
    ---
    ### What is Linda?
    Linda is a next-generation voice-first virtual assistant that enables conversational, human-like interactions in multiple languages.  
    It uses advanced conversational AI to understand intent, emotion, and context, turning customer engagement into smart dialogue and actionable outcomes.
    - **Flexible, natural conversations**: Voice or text, on Web, Mobile, WhatsApp.
    - **Always-on support**: AI-driven insights around the clock.
    - **Brand & domain customization**: Linda learns your unique voice and business needs.
    
    ---
    ### Key Capabilities
    - Voice + Text conversations (Web & WhatsApp)
    - NLP powered by Rasa, Dialogflow, or custom models
    - Voice-to-text via Whisper; smart generative AI responses
    - Deployed anywhere: Web, Mobile, WhatsApp
    - Multilingual & regional language adaptation
    - Brand voice and tone customization
    
    ---
    ### What Makes Linda Different?
    Linda isn’t just a chatbot—it’s a fully AI-enabled virtual assistant:
    - Advanced AI for intent, sentiment, and context understanding
    - Multi-turn conversation support with contextual memory
    - Dynamic, unscripted answers using generative AI and LLM integration
    - Adaptive learning from ongoing user interactions
    - Multimodal NLP (voice, text, user behavior)
    
    ---
    ### Technology Stack
    - Speech Recognition: **OpenAI Whisper**
    - NLP & Dialog Management: **Rasa, Dialogflow, custom transformers**
    - Smart AI Response: **OpenAI/GPT-based models**
    - TTS: **ElevenLabs**, Azure TTS
    - Platform: Web, Mobile, WhatsApp
    - Hosting: On-premise | Private Cloud | Public Cloud
    - Security: End-to-end encryption, role-based access
    
    ---
    ### Real Business Value
    - Reduce support costs by up to **50%**
    - Resolve queries **70% faster** with conversational voice automation
    - Increase conversion rates & customer satisfaction
    - 24/7, multilingual, always-on engagement
    - Seamless integration with CRMs, ERPs, and key business tools
    
    ---
    ### Industries & Use Cases
    | Industry             | Use Cases                                                      |
    |----------------------|---------------------------------------------------------------|
    | Insurance            | Claims, renewals, IVR automation                              |
    | Banking & Finance    | Balance checks, fraud alerts, loan help                       |
    | Healthcare           | Appointments, triage, patient follow-ups                      |
    | Retail & eCommerce   | Order tracking, returns, recommendations                      |
    | Education            | Student Q&A, voice tutors, admissions info                    |
    | Energy & Utilities   | Billing, outages, new service requests                        |
    | Real Estate          | Property info, site visits, lead engagement                   |
    | Travel & Hospitality | Bookings, check-ins, itinerary updates                        |
    
    ---
    ### Customization
    - Multilingual and regional adaptation for diverse audiences
    - Brand voice and tone configuration
    - Domain-specific knowledge and training
    - API and legacy system integration
    
    ---
    ### Linda = Smart Conversations + Tangible Outcomes
    Built by MS Risktec. Powered by AI. Tailored for Your Industry.<br>
    📩 info@msrisktec.com &nbsp;&nbsp; 🌐 www.msrisktec.com
    """, unsafe_allow_html=True)



with tab2:
    st.subheader("Explore Insurance Datasets")
    dataset_choice = st.selectbox("Select Dataset", ["Life/Health", "Motor", " Health"])
    df = df_life if dataset_choice == "Life/Health" else df_motor if dataset_choice == "Motor" else df_synth

    st.dataframe(df.head())

    mode = st.radio("Select mode:", ["Query Dataset", "Run ML Recommender"], key="tab1_mode")

    input_method = st.radio("Choose input method:", ["Text", "Voice"], key="query_input_method")

    if input_method == "Text":
        user_query = st.text_area("Enter your question or customer description here:", key="query_text")
    else:
        if st.button("🎙️ Speak", key="query_speak"):
            user_query = get_voice_input()
        else:
            user_query = ""


    if user_query:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        if mode == "Query Dataset":
            st.write(f"Querying dataset with: {user_query}")
            pandas_code = ask_ai(user_query, df)
            pandas_code = fix_column_names(df, pandas_code)
            st.code(pandas_code, language="python")
            # Try to execute code—if fails, fallback to AI chat
            try:
                result = eval(pandas_code, {"df": df, "pd": pd, "np": np})
                st.write("Result:")
                if isinstance(result, (pd.DataFrame, pd.Series)):
                    st.dataframe(result)
                    # Optional: plot if numeric
                    if isinstance(result, pd.DataFrame) and result.select_dtypes(include=np.number).shape[1] > 0:
                        fig = px.histogram(result, x=result.columns[0])
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write(result)
            except Exception as e:
                st.warning("❓ Could not process as a data query. Switching to AI chat mode ...")
                genai.configure(api_key="AIzaSyADkL6_0GAaUgT9MzWoEGbTJkZfLgWtSE0")
                model = genai.GenerativeModel("models/gemini-2.0-flash")
                chat_response = model.generate_content(
                    f"You are Linda, a helpful AI assistant. Answer the following as clearly as possible for the user: '{user_query}'")
                st.markdown(f"**AI:** {chat_response.text.strip()}")

        elif mode == "Run ML Recommender":
            # st.write("Running product recommendation based on description:")
            # st.write(user_query)
            
            prompt = f"""
        Output ONLY a valid Python dictionary. No explanations, no extra text or comments. Use standard single or double quotes.
        Do not add ```
        You are an assistant extracting insurance customer features from plain English descriptions.
        The dataset columns are: {list(df.drop(columns=['Recommended_Product','RecommendedProduct'], errors='ignore').columns)}.
        Output a valid Python dict with keys matching columns and values extracted from the text:
        \"\"\"{user_query}\"\"\"
        """

            genai.configure(api_key="AIzaSyA0L73IYOdcRIB27Lm3MHsfEEeWE-acyGs")
            model = genai.GenerativeModel("models/gemini-2.0-flash")
            response = model.generate_content(prompt)
            features_dict_str = response.text.strip()

            try:
                features_dict = ast.literal_eval(features_dict_str)
                st.write("Extracted features:", features_dict)
            except Exception:
                features_dict = None
                # No error message displayed here anymore (silent fail)

            # Fallback to AI chat if features_dict is None or empty
            if not features_dict:
                

                chat_prompt = f"You are Linda, a helpful AI assistant. Answer the following question clearly:\n'{user_query}'"
                chat_response = model.generate_content(chat_prompt)
                
                # Larger, more prominent AI response
                st.markdown(f"""
                    <div style='font-size: 1.6em; font-weight: bold; color: #0084ff; padding: 15px; border-radius: 10px; background-color: #e6f0ff;'>
                        {chat_response.text.strip()}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                input_df = prepare_input(features_dict, df, target_col="Recommended_Product" if dataset_choice != "Motor" else "RecommendedProduct")

                recomm = None
                if dataset_choice == "Life/Health":
                    recomm = train_recommender(df_life, target_col="Recommended_Product")
                elif dataset_choice == "Motor":
                    recomm = train_recommender(df_motor, target_col="RecommendedProduct")
                else:
                    recomm = train_recommender(df_synth, target_col="Recommended_Product")

                try:
                    probs = recomm["pipeline"].predict_proba(input_df)
                    pred_idx = np.argmax(probs, axis=1)[0]         # extract scalar index for single sample
                    pred_label = recomm["le"].inverse_transform([pred_idx])[0]  # decode single label

                    st.success(f"✅ Recommended Product: {pred_label}")

                    prob_series = pd.Series(probs[0], index=recomm["le"].classes_).sort_values(ascending=False)
                    st.bar_chart(prob_series)

                    top3_idx = np.argsort(probs[0])[::-1][:3]
                    top3_labels = recomm["le"].inverse_transform(top3_idx)
                    top3_probs = probs[0][top3_idx]

                    st.markdown("### 🔝 Top 3 Recommendations:")
                    for label, prob in zip(top3_labels, top3_probs):
                        st.write(f"- **{label}** : {prob*100:.1f}%")

                except Exception as e:
                    st.error(f"Could not run prediction: {e}")


        # Button to toggle Ask AI Chat visibility
    # if "show_ask_ai" not in st.session_state:
    #     st.session_state.show_ask_ai = False

    # if st.button("Open Ask AI Chat"):
    #     st.session_state.show_ask_ai = True

    # if st.session_state.show_ask_ai:
    #     st.markdown("---")
    #     st.subheader("Linda Ask AI Chat")

    #     # Initialize chat history if needed
    #     if "chat_history" not in st.session_state:
    #         st.session_state.chat_history = []

    #     chat_container = st.container()
    #     with chat_container:
    #         for chat in st.session_state.chat_history:
    #             if chat["role"] == "user":
    #                 st.markdown(f"""
    #                 <div style='text-align: right; margin: 8px'>
    #                     <div style='display: inline-block; background:#0084ff; color:white; padding:8px 12px; border-radius:15px; max-width:65%;'>
    #                         {chat["content"]}
    #                     </div>
    #                 </div>""", unsafe_allow_html=True)
    #             else:
    #                 st.markdown(f"""
    #                 <div style='text-align: left; margin: 8px'>
    #                     <div style='display: inline-block; background:#e5e5ea; color:black; padding:8px 12px; border-radius:15px; max-width:65%;'>
    #                         {chat["content"]}
    #                     </div>
    #                 </div>""", unsafe_allow_html=True)

    #     def send_message():
    #         user_msg = st.session_state.user_input.strip()
    #         if user_msg:
    #             st.session_state.chat_history.append({"role": "user", "content": user_msg})
    #             st.session_state.user_input = ""

    #             genai.configure(api_key="AIzaSyADkL6_0GAaUgT9MzWoEGbTJkZfLgWtSE0")
    #             model = genai.GenerativeModel("models/gemini-2.0-flash")
    #             prompt = f"Answer clearly:\nUser: {user_msg}\nAI:"
    #             response = model.generate_content(prompt)
    #             ai_text = response.text.strip()

    #             st.session_state.chat_history.append({"role": "ai", "content": ai_text})

    #     st.text_input("Type your question and press Enter...", key="user_input", on_change=send_message, placeholder="Ask Linda anything...")

    #     st.markdown("<div style='height:50px'></div>", unsafe_allow_html=True)  # Spacer


    #     st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Utility: Show model metrics
# ---------------------------
def show_model_metrics(recomm):
    metrics = recomm["metrics"]
    st.markdown("### 📈 Model Performance")
    st.table(pd.DataFrame([metrics]))

    # ROC Curve
    fpr, tpr, _ = roc_curve(recomm["y_test"], recomm["y_probs"][:,1], pos_label=1)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC'))
    fig_roc.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash'))
    fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig_roc, use_container_width=True)

    # Feature Importance
    clf = recomm["pipeline"].named_steps["clf"]
    feat_names = recomm["pipeline"].named_steps["preprocessor"].get_feature_names_out()
    importances = clf.feature_importances_
    feat_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False).head(10)
    st.bar_chart(feat_df.set_index("feature"))

# ---------------------------
# 🏥 Tab 2: Life/Health Recommender
# ---------------------------
with tab3:
    st.header("🏥 Life/Health Insurance Product Recommendation")
    st.dataframe(df_life.head())

    recomm = train_recommender(df_life, target_col="Recommended_Product")

    sample = collect_user_inputs(df_life, exclude_cols=["Recommended_Product"], dataset_name="life")
    st.dataframe(sample)

    try:
        user_df = prepare_input(sample.iloc[0].to_dict(), df_life, target_col="Recommended_Product")
        probs = recomm["pipeline"].predict_proba(user_df)
        pred_idx = np.argmax(probs, axis=1)[0]
        pred_label = recomm["le"].inverse_transform([pred_idx])[0]

        st.success(f"✅ Recommended Product: {pred_label}")

# Bar chart for all class probabilities
        prob_series = pd.Series(probs[0], index=recomm["le"].classes_).sort_values(ascending=False)
        st.bar_chart(prob_series)

        # Show Top 3 recommendations with % values
        top3_idx = np.argsort(probs[0])[::-1][:3]
        top3_labels = recomm["le"].inverse_transform(top3_idx)
        top3_probs = probs[0][top3_idx]

        st.markdown("### 🔝 Top 3 Recommendations:")
        for label, prob in zip(top3_labels, top3_probs):
            st.write(f"- **{label}** : {prob*100:.1f}%")

    except Exception as e:
        st.error(f"Could not predict: {e}")

    # show_model_metrics(recomm)

# ---------------------------
# 🚗 Tab 3: Motor Recommender
# ---------------------------
with tab4:
    st.header("🚗 Motor Insurance Product Recommendation")
    st.dataframe(df_motor.head())

    recomm_motor = train_recommender(df_motor, target_col="RecommendedProduct")

    sample_motor = collect_user_inputs(df_motor, exclude_cols=["RecommendedProduct"], dataset_name="motor")
    st.dataframe(sample_motor)

    try:
        user_df_motor = prepare_input(sample_motor.iloc[0].to_dict(), df_motor, target_col="RecommendedProduct")
        probs_motor = recomm_motor["pipeline"].predict_proba(user_df_motor)
        pred_idx_motor = np.argmax(probs_motor, axis=1)[0]
        pred_label_motor = recomm_motor["le"].inverse_transform([pred_idx_motor])[0]

        st.success(f"✅ Recommended Product: {pred_label}")

        # Bar chart for all class probabilities
        prob_series = pd.Series(probs[0], index=recomm["le"].classes_).sort_values(ascending=False)
        st.bar_chart(prob_series)

        # Show Top 3 recommendations with % values
        top3_idx = np.argsort(probs[0])[::-1][:3]
        top3_labels = recomm["le"].inverse_transform(top3_idx)
        top3_probs = probs[0][top3_idx]

        st.markdown("### 🔝 Top 3 Recommendations:")
        for label, prob in zip(top3_labels, top3_probs):
            st.write(f"- **{label}** : {prob*100:.1f}%")

    except Exception as e:
        st.error(f"Could not predict: {e}")

    # show_model_metrics(recomm_motor)

# ---------------------------
# 🧪 Tab 4: Synthetic Health Recommender
# ---------------------------
with tab5:
    st.header("🧪 Health Insurance Product Recommendation")
    st.dataframe(df_synth.head())

    recomm_synth = train_recommender(df_synth, target_col="Recommended_Product")

    sample_synth = collect_user_inputs(df_synth, exclude_cols=["Recommended_Product"], dataset_name="synth")
    st.dataframe(sample_synth)

    try:
        user_df_synth = prepare_input(sample_synth.iloc[0].to_dict(), df_synth, target_col="Recommended_Product")
        probs_synth = recomm_synth["pipeline"].predict_proba(user_df_synth)
        pred_idx_synth = np.argmax(probs_synth, axis=1)[0]
        pred_label_synth = recomm_synth["le"].inverse_transform([pred_idx_synth])[0]

        st.success(f"✅ Recommended Product: {pred_label}")

# Bar chart for all class probabilities
        prob_series = pd.Series(probs[0], index=recomm["le"].classes_).sort_values(ascending=False)
        st.bar_chart(prob_series)

        # Show Top 3 recommendations with % values
        top3_idx = np.argsort(probs[0])[::-1][:3]
        top3_labels = recomm["le"].inverse_transform(top3_idx)
        top3_probs = probs[0][top3_idx]

        st.markdown("### 🔝 Top 3 Recommendations:")
        for label, prob in zip(top3_labels, top3_probs):
            st.write(f"- **{label}** : {prob*100:.1f}%")

    except Exception as e:
        st.error(f"Could not predict: {e}")

    # show_model_metrics(recomm_synth)

