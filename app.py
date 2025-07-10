import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set page config
st.set_page_config(
    page_title="🚢 Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🚢 Titanic Survival Predictor")
st.markdown("""
Welcome to the **Titanic Survival Predictor**! 🌊

Enter passenger details to predict the chances of survival. Explore the Titanic dataset, visualize insights, and learn more about the tragedy. 

---
""")

# Sidebar for navigation
st.sidebar.title("🔎 Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "Dataset", "Visualizations", "About"])

# Load dataset (sample or from URL)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# Preprocessing for model
@st.cache_data
def preprocess_data(df):
    data = df.copy()
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    data = data.fillna(data.median(numeric_only=True))
    data = data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
    return data

# Train model
@st.cache_resource
def train_model(data):
    X = data.drop('Survived', axis=1)
    y = data['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc

preprocessed = preprocess_data(df)
model, model_acc = train_model(preprocessed)

# Prediction Page
if page == "Prediction":
    st.header("🧑‍💼 Passenger Details")
    col1, col2 = st.columns(2)
    with col1:
        pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3], index=0)
        sex = st.radio("Sex", ["male", "female"])
        age = st.slider("Age", 0, 80, 25)
        sibsp = st.number_input("# Siblings/Spouses Aboard", 0, 8, 0)
    with col2:
        parch = st.number_input("# Parents/Children Aboard", 0, 6, 0)
        fare = st.number_input("Fare ($)", 0.0, 600.0, 32.2)
        embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"], index=0)

    st.markdown("---")
    if st.button("🚀 Predict Survival", use_container_width=True):
        input_data = pd.DataFrame({
            'Pclass': [pclass],
            'Sex': [0 if sex == 'male' else 1],
            'Age': [age],
            'SibSp': [sibsp],
            'Parch': [parch],
            'Fare': [fare],
            'Embarked': [ {'S':0, 'C':1, 'Q':2}[embarked] ]
        })
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
        if prediction == 1:
            st.success(f"🎉 Survived! Probability: {prob:.2%}")
        else:
            st.error(f"💀 Did not survive. Probability: {prob:.2%}")
        st.info(f"Model Accuracy: {model_acc:.2%}")

# Dataset Page
elif page == "Dataset":
    st.header("📊 Titanic Dataset Preview")
    st.dataframe(df.head(50), use_container_width=True)
    st.markdown(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
    st.markdown("---")
    st.markdown("[Download Full Dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)")

# Visualizations Page
elif page == "Visualizations":
    import matplotlib.pyplot as plt
    import seaborn as sns
    st.header("📈 Data Visualizations")
    st.subheader("Survival by Class")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='Pclass', hue='Survived', data=df, ax=ax1, palette='Set2')
    ax1.set_xticklabels(['1st', '2nd', '3rd'])
    st.pyplot(fig1)

    st.subheader("Survival by Sex")
    fig2, ax2 = plt.subplots()
    sns.countplot(x='Sex', hue='Survived', data=df, ax=ax2, palette='Set1')
    st.pyplot(fig2)

    st.subheader("Age Distribution")
    fig3, ax3 = plt.subplots()
    sns.histplot(df['Age'].dropna(), kde=True, color='skyblue', bins=30)
    st.pyplot(fig3)

    st.subheader("Fare Distribution")
    fig4, ax4 = plt.subplots()
    sns.histplot(df['Fare'], kde=True, color='gold', bins=30)
    st.pyplot(fig4)

# About Page
elif page == "About":
    st.header("ℹ️ About This App")
    st.markdown("""
    - **Project:** Titanic Survival Prediction
    - **Model:** Random Forest Classifier
    - **Dataset:** [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)
    - **Features:**
        - Predict survival chances
        - Explore and visualize the dataset
        - Attractive UI with emojis and colors
        - Model accuracy display
        - Download dataset
    - **Author:** Your Name 🚀
    
    ---
    **The Titanic disaster** occurred in 1912 when the RMS Titanic sank after hitting an iceberg. This app uses machine learning to predict survival chances based on passenger data.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", caption="RMS Titanic", use_column_width=True)
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit.") 