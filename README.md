# ❤️ Heart Disease Prediction Model

This project aims to predict the likelihood of heart disease in a patient using multiple machine learning algorithms. The dataset contains medical attributes such as age, sex, cholesterol levels, blood pressure, and more.  
The model helps in early detection and preventive healthcare by analyzing patient data.

---

## 🚀 Features
- Predicts whether a person has heart disease (1) or not (0)
- Compares performance across multiple ML algorithms:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Decision Tree
  - Naive Bayes
- Shows accuracy comparison of all models in one place
- Easy to extend and deploy (e.g., with Streamlit or Flask)

---

## 📂 Dataset
The dataset used is the **Heart Disease Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease).  
It includes features like:
- Age  
- Sex  
- Chest Pain Type (cp)  
- Resting Blood Pressure (trestbps)  
- Serum Cholesterol (chol)  
- Fasting Blood Sugar (fbs)  
- Resting ECG Results (restecg)  
- Maximum Heart Rate Achieved (thalach)  
- Exercise Induced Angina (exang)  
- Oldpeak (ST depression)  
- Slope, CA, Thal  
- Target (0 = No Disease, 1 = Disease)

---

## 🧠 Machine Learning Models Used
| Model | Description | Accuracy (example) |
|--------|--------------|--------------------|
| Logistic Regression | Linear model for classification | 85% |
| K-Nearest Neighbors | Distance-based non-parametric model | 83% |
| SVM | Finds optimal hyperplane for separation | 86% |
| Decision Tree | Tree-based classification | 81% |
| Naive Bayes | Probabilistic model based on Bayes theorem | 82% |

*(You can replace the accuracy values with your own results.)*

---

## ⚙️ Workflow
1. **Data Preprocessing**
   - Handling missing values
   - Encoding categorical data
   - Feature scaling

2. **EDA (Exploratory Data Analysis)**
   - Correlation heatmap
   - Feature importance
   - Outlier detection

3. **Model Training**
   - Train/test split (e.g., 80/20)
   - Train all five models

4. **Model Evaluation**
   - Accuracy, precision, recall, F1-score
   - Compare model results in one table

5. **Model Saving**
   - Use `pickle` to save the best model

---

## 💾 Example Code Snippet

```python
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
with open('heart_model.pkl', 'wb') as f:
    pickle.dump(model, f)
