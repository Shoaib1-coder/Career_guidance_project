#  Import required libraries
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import string
import plotly.graph_objs as go
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# -------------------- LOAD STYLES --------------------
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -------------------- LOAD MODEL & DATA --------------------
@st.cache_resource
def load_assets():
    model = joblib.load("intent_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    df = pd.read_csv("updated_with_all.csv")
    df.columns = df.columns.str.strip().str.lower()
    df['role'] = df['role'].str.strip().str.title()
    return model, vectorizer, df

model, vectorizer, df = load_assets()

# -------------------- HEADER --------------------
st.markdown("""
<div class="header">
    <h1 style="text-align: center;">🎓 Career Guidance Chatbot</h1>
    <p style="text-align: center; opacity: 0.9;">Discover your perfect career match based on your interests</p>
</div>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("## 🔍 Explore Careers")
    popular_roles = df['role'].value_counts().index[:5]
    for role in popular_roles:
        if st.button(role, key=f"btn_{role}"):
            st.session_state.user_input = f"I'm interested in {role.lower()}"
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.info("Try:\n- What is the job description for an AI Researcher?\n- What does a QA Engineer do?\n- A typical day for a Data Analyst?")

# -------------------- INPUT --------------------
col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_input("Describe your interests or skills:", placeholder="e.g., I enjoy solving problems...", key="user_input")
with col2:
    suggest_btn = st.button("🎯 Career Suggestion")

# -------------------- SESSION STATE INIT --------------------
if 'predicted_role' not in st.session_state:
    st.session_state.predicted_role = None
if 'response_row' not in st.session_state:
    st.session_state.response_row = None
if 'top_roles' not in st.session_state:
    st.session_state.top_roles = None

# -------------------- MAIN LOGIC --------------------
if suggest_btn and user_input:
    with st.spinner("🔮 Analyzing your interests..."):
        cleaned_input = user_input.lower().translate(str.maketrans('', '', string.punctuation))
        input_vector = vectorizer.transform([cleaned_input])
        predicted_role = model.predict(input_vector)[0].title()
        st.session_state.predicted_role = predicted_role

        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(input_vector)[0]
            top_roles = pd.DataFrame({'Role': model.classes_, 'Confidence': probas}).sort_values('Confidence', ascending=False).head(3)
            st.session_state.top_roles = top_roles

        response_row = df[df['role'] == predicted_role]
        st.session_state.response_row = response_row if not response_row.empty else None

# -------------------- DISPLAY RESULTS --------------------
if st.session_state.predicted_role and st.session_state.response_row is not None:
    response_row = st.session_state.response_row
    predicted_role = st.session_state.predicted_role
    top_roles = st.session_state.top_roles

    match_found = any(word in user_input.lower() for word in predicted_role.lower().split())

    if predicted_role in df['role'].values and match_found:
        row = df[df['role'] == predicted_role]
        desc = row['answer'].values[0] if row['answer'].notnull().values[0] else "No description available."

        st.markdown(f"""
        <div class="card" style="font-size:1rem;">
            <div class="role-badge">✨ Recommended Career</div>
            <h2 style="color:blue;">{predicted_role}</h2>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

        # Show confidence bars
        if top_roles is not None and not top_roles.empty:
            st.markdown("#### Other potential matches:")
            for _, row in top_roles.iterrows():
                progress = int(row['Confidence'] * 100)
                st.markdown(f"""
                <div style="margin: 0.5rem 0;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>{row['Role'].title()}</span>
                        <span>{progress}%</span>
                    </div>
                    <progress value="{progress}" max="100" style="width: 100%; height: 8px;"></progress>
                </div>
                """, unsafe_allow_html=True)

        # Toggle flags
        st.session_state.setdefault('show_salary', False)
        st.session_state.setdefault('show_education', False)
        st.session_state.setdefault('show_courses', False)

        # Buttons
        st.markdown("#### 📚 Learn More About This Career")
        if st.button("💰 Salary Range"):
            st.session_state.show_salary = not st.session_state.show_salary
        if st.button("🎓 Required Education"):
            st.session_state.show_education = not st.session_state.show_education
        if st.button("📘 Courses"):
            st.session_state.show_courses = not st.session_state.show_courses

        # Show extra info
        if st.session_state.show_salary:
            
            st.markdown(f"""
            <div class='card'>
                <div class='role-badge'>🎓 Salary Per Month </div>
                <p style="font-size: 1rem; color: blue;">{response_row['salary_range'].values[0]}</p>
            </div>
            """, unsafe_allow_html=True)

        if st.session_state.show_education:
            st.markdown(f"""
            <div class='card'>
                <div class='role-badge'>🎓 Education</div>
                <p style="font-size: 1rem; color: blue;">{response_row['education'].values[0]}</p>
                </div>
            """, unsafe_allow_html=True)

        if st.session_state.show_courses:
          st.markdown(f"""
          <div class='card'>
           <div class='role-badge'>📘 Courses</div>
           <p style="font-size: 1rem; color: blue;">{response_row['courses_name'].values[0]}</p>
          </div>
          """, unsafe_allow_html=True)

    else:
        st.warning(f"🚫 Sorry, we couldn't find a career matching **'{user_input}'**. Not train model on this Career Try rephrasing or adding more details.")

# -------------------- NO INPUT --------------------
elif not user_input:
    st.markdown("""
    <div style="text-align: center; margin-top: 3rem; opacity: 0.6; color: red;">
        <p class="typing-animation">Waiting for your career interests...</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------- MODEL PERFORMANCE --------------------
st.markdown("## 📊 Model Performance (SVM)")
df1 = pd.read_csv("career_guidance_dataset.csv")
df1['question'] = df1['question'].apply(lambda x: x.lower().translate(str.maketrans('', '', string.punctuation)))
X = vectorizer.transform(df1['question'])
y = df1['role']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
# Bar plot data
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
scores = [acc, prec, rec, f1]

# Create bar plot
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(metrics, scores, color=['red', 'blue', 'black', 'yellow'])

# Add score labels above bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center', fontsize=10)

# Customize plot
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score")
ax.set_title("Model Metrics")
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Display in Streamlit
st.pyplot(fig)


# Confusion matrix
st.markdown("### 📌 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
fig_cm = go.Figure(data=go.Heatmap(z=cm, x=model.classes_, y=model.classes_, colorscale='Viridis'))
fig_cm.update_layout(title='Confusion Matrix', xaxis_title="Predicted", yaxis_title="Actual")
st.plotly_chart(fig_cm)

# -------------------- SUMMARY STATS --------------------
st.markdown("## 📈 Dataset & Model Info")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown("<div class='metric-box'><h3>🎯 Unique Careers</h3><p>{}</p></div>".format(df['role'].nunique()), unsafe_allow_html=True)
with col2:
    st.markdown("<div class='metric-box'><h3>📄 Total Questions</h3><p>{}</p></div>".format(df.shape[0]), unsafe_allow_html=True)
with col3:
    st.markdown("<div class='metric-box'><h3 >🧠 Model</h3><p>Support Vector Machine Algorithm</p></div>", unsafe_allow_html=True)


# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Career Guidance • Powered by Shoaib • Updated@2025</p>
</div>
""", unsafe_allow_html=True)
