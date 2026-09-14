# 🏦 Customer Churn Prediction

### Predicting bank customer churn with classical ML + deep learning, deployed as a live web app

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Models-orange?logo=scikitlearn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras%20ANN-FF6F00?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed%20App-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**[🚀 Live Demo](#-deployment)** · **[📊 Model Results](#-models-trained--compared)** · **[⚙️ Run Locally](#run-locally)**

---

## 📌 Overview

Customer churn is costly for banks — retaining an existing customer is far cheaper than acquiring a new one. This project builds and compares **6 classification models** to flag customers at high risk of churning, so a business can act proactively with targeted retention offers.

The best-performing model (an Artificial Neural Network) is deployed as an **interactive Streamlit app**, so anyone can enter a customer's profile and get an instant churn prediction.

*Data Cleaning → Feature Engineering → Model Training → Evaluation → Deployment*

---

## 📊 Dataset

The dataset contains **~10,000 bank customer records** with the following features:

| Feature | Description |
|---|---|
| `CreditScore` | Customer's credit score |
| `Geography` | Country (France, Germany, Spain) |
| `Gender` | Male / Female |
| `Age` | Customer's age |
| `Tenure` | Years as a bank customer |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products used |
| `HasCrCard` | Whether the customer has a credit card |
| `IsActiveMember` | Whether the customer is an active member |
| `EstimatedSalary` | Estimated annual salary |
| `Exited` | 🎯 **Target** — `1` if the customer churned, `0` otherwise |

> `RowNumber`, `CustomerId`, and `Surname` were dropped — pure identifiers with no predictive value.

---

## 🛠 Feature Engineering & Preprocessing

**Two engineered features** to surface relationships the raw columns don't capture directly:

| New Feature | Formula |
|---|---|
| `balance_to_salary` | `Balance / EstimatedSalary` |
| `Tenure_by_age` | `Balance / Age` |

**Encoding:**
- `Gender` → Label encoded (Female = 0, Male = 1)
- `Geography` → One-hot encoded → `Geography_Germany`, `Geography_Spain`

**Final feature set (13 columns)** fed into every model, in this exact order:
```
CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard,
IsActiveMember, EstimatedSalary, balance_to_salary, Tenure_by_age,
Geography_Germany, Geography_Spain
```

**Split & Scale:** 80/20 train-test split (`random_state=12`) → `StandardScaler` fit on training data only.

---

## 🤖 Models Trained & Compared

Six classification models, same train/test split, same preprocessing:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.8095 | 0.6871 | 0.2317 | 0.3465 |
| KNN Classifier | 0.8290 | 0.7009 | 0.3761 | 0.4896 |
| Naive Bayes | 0.7890 | 0.7692 | 0.0459 | 0.0866 |
| Random Forest | 0.8675 | 0.8327 | 0.4908 | 0.6176 |
| XGBoost Classifier | 0.8540 | 0.7416 | 0.5069 | 0.6022 |
| 🏆 **ANN Model (deployed)** | **0.8575** | **0.8032** | **0.4587** | **0.5839** |

> The **ANN** was selected for deployment for its strong precision and balanced overall performance, and to showcase a complete deep-learning deployment pipeline.

### 🧠 ANN Architecture

```
Input (13 features)
   │
   ▼
Dense(12, activation='relu')
   │
   ▼
Dense(6, activation='relu')
   │
   ▼
Dense(1, activation='sigmoid')  →  Churn Probability
```

Compiled with `optimizer='adam'`, `loss='binary_crossentropy'` · trained for **100 epochs**, batch size **32**, `validation_split=0.2`.

Confusion matrices were plotted for every model to visualize prediction errors alongside the metrics above.

---

## 🚀 Deployment

The ANN model is deployed as an interactive **Streamlit** app — enter a customer's profile and get an instant churn prediction with probability.

### Run locally

```bash
git clone https://github.com/VidyaBind/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```
churn-prediction/
│
├── app.py                     # Streamlit app for live predictions
├── requirements.txt           # Python dependencies
├── churn_ann_model.h5         # Saved trained ANN model
├── scaler.pkl                 # Saved StandardScaler (for preprocessing new inputs)
├── notebook.ipynb             # Full analysis: EDA, feature engineering, model training & comparison
├── data/
│   └── Churn_Modelling.csv    # Raw dataset
└── README.md                  # You are here
```

---

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python |
| **Data Handling** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **Classical ML** | scikit-learn (Logistic Regression, KNN, Naive Bayes, Random Forest), XGBoost |
| **Deep Learning** | TensorFlow / Keras |
| **Deployment** | Streamlit |

---

## 📈 Key Insights

- 🔗 Most raw features showed weak linear correlation with churn — `Age` had the strongest positive correlation among the original columns.
- 💰 Customers with fewer products and higher balance-to-salary ratios appear more prone to churn.
- 🌳 Ensemble/non-linear models (Random Forest, XGBoost, ANN) clearly outperformed simpler linear models on recall and F1 — churn isn't well captured by linear relationships alone.
- ⚖️ All models struggle more with **recall** than precision — a notable share of actual churners are still missed, a common challenge with imbalanced classes.

---

## 🔮 Future Improvements

- [ ] Hyperparameter tuning (GridSearchCV / Keras Tuner) to boost recall
- [ ] Handle class imbalance with SMOTE or class weighting
- [ ] Add SHAP values for model explainability
- [ ] Migrate to a production backend (FastAPI/Flask) with prediction logging

---

## 👤 Author

**Vidya Bind**

📧 vidyabind50@gmail.com

