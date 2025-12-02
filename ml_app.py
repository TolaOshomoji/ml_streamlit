# PART 2 (35 points): RandomForest Predictor for Classification of flower species

# 1. Import the following libraries.
    
import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


st.title("Tola's Iris Flower Species Predictor")
st.subheader("RandomForest classifier using the classic Iris dataset")

# 2. Train a RandomForest Machine Learning model for the given dataset.

# Cache data for efficient loading
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target
    return df, iris.target_names


# Load data
df, target_name = load_data()

st.write("### Training data preview")
st.dataframe(df.head())

# Dataset for model training
X = df.iloc[:, :-1]      # all feature columns
y = df["species"]        # target labels

# Model training for given dataset
model = RandomForestClassifier()
model.fit(X, y)

# 3. Generate test samples from user inputs. Predict the flower species based for the given input (sepal length, sepal width, petal length and petal width)

# Generate test samples from user inputs
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame({
        'sepal length (cm)': [],
        'sepal width (cm)' : [],
        'petal length (cm)': [],
        'petal width (cm)' : [],
        'predicted species': []
    })

st.write("### Enter values for input features:")

with st.form("add_row_form"):
    sepal_length = st.slider(
        "Sepal length (cm)",
        float(df['sepal length (cm)'].min()),
        float(df['sepal length (cm)'].max()),
        float(df['sepal length (cm)'].mean())
    )

    sepal_width  = st.slider(
        "Sepal width (cm)",
        float(df['sepal width (cm)'].min()),
        float(df['sepal width (cm)'].max()),
        float(df['sepal width (cm)'].mean())
    )

    petal_length = st.slider(
        "Petal length (cm)",
        float(df['petal length (cm)'].min()),
        float(df['petal length (cm)'].max()),
        float(df['petal length (cm)'].mean())
    )

    petal_width  = st.slider(
        "Petal width (cm)",
        float(df['petal width (cm)'].min()),
        float(df['petal width (cm)'].max()),
        float(df['petal width (cm)'].mean())
    )

    submitted = st.form_submit_button("Predict")

    if submitted:
        # Prediction            
        input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = model.predict(input_data)
        predicted_species = target_name[prediction[0]]

        new_row = pd.DataFrame([{
            'sepal length (cm)': sepal_length,
            'sepal width (cm)' : sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)': petal_width,
            'predicted species': predicted_species
        }])
        
        st.session_state.df = pd.concat(
            [st.session_state.df, new_row],
            ignore_index=True
        )
        
        st.success(f"Row added!   Predicted species is {prediction[0]} – {predicted_species}")

st.write("### Predicted species for tested samples:")
st.dataframe(st.session_state.df)
