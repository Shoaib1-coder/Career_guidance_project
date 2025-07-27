# Import necessary libraries
import pandas as pd  # for data handling
import string  # for text preprocessing (removing punctuation)
import joblib  # to save/load model and vectorizer
from sklearn.model_selection import train_test_split  # to split data
from sklearn.feature_extraction.text import TfidfVectorizer  # for text feature extraction
from sklearn.svm import SVC  # Support Vector Classifier

# -------------------- Load Dataset --------------------
df = pd.read_csv("career_guidance_dataset.csv")  # Load dataset from CSV file

# -------------------- Text Cleaning Function --------------------
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

# Apply cleaning to the 'question' column
df['cleaned_question'] = df['question'].apply(clean_text)

# -------------------- TF-IDF Vectorization --------------------
vectorizer = TfidfVectorizer()  # Create TF-IDF vectorizer object
X = vectorizer.fit_transform(df['cleaned_question'])  # Fit and transform the cleaned text
y = df['role']  # Target variable is the career role

# -------------------- Train-Test Split --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)  # Split data into training and testing sets

# -------------------- Train SVM Model --------------------
model = SVC(kernel='linear', probability=True)  # Create SVM model with linear kernel
model.fit(X_train, y_train)  # Train the model on training data

# -------------------- Save Model and Vectorizer --------------------
joblib.dump(model, "intent_model.pkl")  # Save the trained model to a file
joblib.dump(vectorizer, "vectorizer.pkl")  # Save the TF-IDF vectorizer to a file

