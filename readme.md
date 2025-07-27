# 🎓 Career Guidance Chatbot

A machine learning-based Career Guidance Chatbot that helps users discover potential career paths based on their interests and questions. Built using Python, Scikit-learn, Streamlit, and trained with an SVM model on a custom career dataset.

---

## 🚀 Features

- Clean and preprocess user queries  
- Vectorize input using TF-IDF  
- Classify into career roles using Support Vector Machine (SVM)  
- Interactive web interface with Streamlit  
- Display top career match and suggestions  
- Realtime feedback for unmatched queries  

---

## 🛠️ Installation Guide (Using Anaconda)

### ✅ Step 1: Download Anaconda (if not already installed)

🔗 [Download Anaconda](https://www.anaconda.com/products/distribution)

---

### ✅ Step 2: Create a virtual environment


conda create -n careerbot python=3.9
conda activate careerbot
### ✅ Step 3: Clone the repository

git clone https://github.com/shoaib1-coder/Career_guidance_project.git
cd Career_guidance_project
### ✅ Step 4: Install dependencies

pip install -r requirements.txt
OR, manually install:


pip install streamlit pandas scikit-learn joblib
▶️ Run the App

streamlit run app.py

### 📁 Project Structure

Career_guidance_project/
│
├── app.py # Streamlit app
├── train_model.py # Train the model
├── career_guidance_dataset.csv
├── updated_with_all.csv
├── intent_model.pkl # Trained SVM model
├── vectorizer.pkl # TF-IDF vectorizer
├── README.md
└── requirements.txt

### 📬 Contact

##### 👨‍💻 SHoaib Sattar 

🔗 www.linkedin.com/in/shoaib-0b64a2204

📧 shoaibmachinelearning@gmail.com

🧠 https://github.com/shoaib1-coder

📜 License
This project is open-source for educational purposes
