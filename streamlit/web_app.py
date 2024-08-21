import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

st.set_page_config(
    page_title="Credit Score Project",
    page_icon="https://cdn-icons-png.flaticon.com/512/1604/1604695.png"
)

# Define helper functions
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_data(df):
    return df

def plot_data(df):
    st.line_chart(df)

def load_model():
    with open('../models/lr_pipeline.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def evaluate_model(model, df, target_column):
    X = df.drop(target_column, axis=1)  # Use the correct target column name
    y_true = df[target_column]  # Use the correct target column name

    # Ensure X is a DataFrame before passing it to the model
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=df.drop(target_column, axis=1).columns)

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_proba),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred)
    }
    return metrics

def main():

    # Sidebar navigation
    page = st.sidebar.selectbox("Navigate", ["Home", "Important Resources"])

    if page == "Home":
        # Home page content
        st.markdown(
            """
            <h1 style='text-align: center;'>
                <img src="https://cdn-icons-png.flaticon.com/512/1604/1604695.png" alt="icon" width="50"/>
                Credit Score Project
            </h1>
            """, unsafe_allow_html=True
        )

        st.markdown(
            """
            <p style="text-align: center; font-size: 22px;">This web application has the goal of allowing easy access to the credit default model trained on the Jupyter-Notebook attached to the <a href="https://github.com/PedroHang/Credit-Scoring-Project" target="_blank">GitHub Repository</a> of this project. For more information, check out the "Important Resources" page of this application located on the sidebar</p>
            """, unsafe_allow_html=True
        )

        st.sidebar.header("User Input")
        st.sidebar.markdown("You can enter the <b>credit_scoring_web.csv</b> file located in the /data folder of the github repository", unsafe_allow_html=True)

        # File uploader
        uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

        if uploaded_file is not None:
            # Load and preprocess data
            df = load_data(uploaded_file)
            df = preprocess_data(df)

            # Slider to select number of rows to display
            num_rows = st.slider('Select the number of rows to display:', min_value=5, max_value=len(df), value=5)

            # Show data
            st.subheader("Raw Data")
            st.write(df.head(num_rows)) 

            # Load pre-trained model
            model = load_model()

            # Define your actual target column name here
            target_column = "default"  # Replace with your actual target column name

            # Evaluate the model
            metrics = evaluate_model(model, df, target_column)
            
            # Display evaluation metrics
            st.subheader("Model Evaluation Metrics")
            st.write(f"**Accuracy:** {metrics['Accuracy']:.2f}")
            st.write(f"**AUC:** {metrics['AUC']:.2f}")
            st.write(f"**Precision:** {metrics['Precision']:.2f}")
            st.write(f"**Recall:** {metrics['Recall']:.2f}")
            st.write(f"**F1 Score:** {metrics['F1 Score']:.2f}")

            # Generate and display plots
            st.subheader("Data Visualization")
            plot_data(df)
        else:
            # Warning message with custom styling
            st.markdown(
                """
                <div style="display: flex; justify-content: center; align-items: center; height: 200px;">
                    <div style="text-align: center; background-color: #FFDDC1; padding: 20px; border-radius: 10px; border: 2px solid #FF7D00;">
                        <h3 style="color: #78290F;">⚠️ Please upload a CSV file to proceed.</h3>
                    </div>
                </div>
                """, unsafe_allow_html=True
            )

    elif page == "Important Resources":
        # Important Resources page content
        st.markdown(
            """
            <h1 style='text-align: center;'>
                <img src="https://cdn-icons-png.flaticon.com/512/4340/4340234.png" alt="icon" width="50"/>
                Important Resources
            </h1>
            <p style="text-align: center; font-size: 22px;">
                This project is multifaceted, drawing on a diverse range of tools and techniques that are 
                <b>CRUCIAL</b> for ensuring thorough analysis and clear communication. These resources are integral to 
                the project's success, providing the foundation for robust and insightful results. Key resources include:
            </p>
            """, unsafe_allow_html=True
        )

        # Create side-by-side buttons with enhanced styling
        st.markdown(
            """
            <div style="display: flex; justify-content: space-around; margin-top: 20px;">
                <a href="https://youtube.com/your-video-link" target="_blank" style="background-color: #cc0000; color: white; padding: 16px 32px; text-decoration: none; border-radius: 8px; font-weight: bold; display: inline-block; width: 20%; text-align: center;">YouTube Video</a>
                <a href="https://kaggle.com/your-project-link" target="_blank" style="background-color: #0000cc; color: white; padding: 16px 32px; text-decoration: none; border-radius: 8px; font-weight: bold; display: inline-block; width: 20%; text-align: center;">Kaggle Project</a>
                <a href="https://github.com/PedroHang/Credit-Scoring-Project" target="_blank" style="background-color: #660066; color: white; padding: 16px 32px; text-decoration: none; border-radius: 8px; font-weight: bold; display: inline-block; width: 20%; text-align: center;">GitHub Repository</a>
                <a href="https://powerbi.com/your-dashboard-link" target="_blank" style="background-color: #cc6600; color: white; padding: 16px 32px; text-decoration: none; border-radius: 8px; font-weight: bold; display: inline-block; width: 20%; text-align: center;">PowerBI Dashboard</a>
            </div>
            """, unsafe_allow_html=True
        )

        st.markdown(
            """
            <p style="text-align: center; margin-top: 40px;">Check out my LinkedIn page:</p>
            <div style="display: flex; justify-content: center;">
                <a href="https://linkedin.com/in/your-profile-link" target="_blank" style="background-color: #0077cc; color: white; padding: 16px 32px; text-decoration: none; border-radius: 8px; font-weight: bold; display: inline-block; text-align: center;">LinkedIn Page</a>
            </div>
            """, unsafe_allow_html=True
        )

# Run the app
if __name__ == "__main__":
    main()
