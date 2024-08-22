import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pycaret.classification import load_model, predict_model
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve

st.set_page_config(
    page_title="Credit Score Project",
    page_icon="https://cdn-icons-png.flaticon.com/512/1604/1604695.png"
)

# Define helper functions with caching
@st.cache_data
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

@st.cache_data
def preprocess_data(df):
    # Ensure 'reference_date' is a datetime64 dtype
    df['reference_date'] = pd.to_datetime(df['reference_date'])
    return df

def plot_auc(y_true, y_proba):
    # Calculate the ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    
    # Plot the AUC
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='orange', lw=2, label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    st.pyplot(plt)

@st.cache_resource
def load_trained_model():
    # Load the model using PyCaret
    return load_model('../models/lr_pipeline')

def evaluate_model(df):
    model = load_trained_model()

    # Use PyCaret's predict_model function to generate predictions and metrics
    predictions = predict_model(model, data=df)
    
    # Extract true labels and predicted probabilities
    y_true = predictions['default']  # Adjust this to your actual target column
    y_pred = predictions['prediction_label']  # Adjust this if necessary
    y_proba = predictions['prediction_score']  # Adjust this if necessary

    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_proba),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred)
    }

    return metrics, y_true, y_proba

def main():
    # Sidebar navigation
    page = st.sidebar.selectbox("Navigate", ["Home", "Important Resources", "Dashboard"])

    if page == "Home":
        # Home page content
        st.markdown(
            """
            <h1 style='text-align: center;'>
                <img src="https://cdn-icons-png.flaticon.com/512/1604/1604695.png" alt="icon" width="50"/>
                Credit Score Project
            </h1>
            <h6 style='text-align: center;'>- Pedro Hang -</h6>
            <hr style="border: 1px solid #ff7d00;">
            """, unsafe_allow_html=True
        )

        st.markdown(
            """
            <p style="text-align: center; font-size: 22px;">
                This web application has the goal of allowing easy access to the credit default model 
                trained on the Jupyter-Notebook attached to the 
                <a href="https://github.com/PedroHang/Credit-Scoring-Project" target="_blank">GitHub Repository</a> 
                of this project. For more information, check out the "Important Resources" page of this 
                application located on the sidebar.
            </p>
            <hr style="border: 1px solid #ff7d00;">
            """, unsafe_allow_html=True
        )

        st.sidebar.header("User Input")
        st.sidebar.markdown("You can enter the <b>credit_scoring_web.csv</b> file located in the /data folder of the GitHub repository", unsafe_allow_html=True)

        # File uploader
        uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

        if uploaded_file is not None:
            # Load and preprocess data
            df = load_data(uploaded_file)
            df = preprocess_data(df)  # Convert 'reference_date' to datetime

            # Slider to select number of rows to display
            num_rows = st.slider('Select the number of rows to display:', min_value=5, max_value=len(df), value=5)

            # Show data
            st.subheader("Raw Data")
            st.write(df.head(num_rows)) 
            st.markdown("<hr style='border: 1px solid #ff7d00;'>", unsafe_allow_html=True)

            # Evaluate the model and display metrics
            metrics, y_true, y_proba = evaluate_model(df)
            
            # Display evaluation metrics with matching text color
            st.markdown(
                """
                <div style="background-color: #ffecd1; padding: 20px; border-radius: 10px;">
                    <h2 style="color: #78290F;">📊 Model Evaluation Metrics</h2>
                    <p style="color: #78290F;"><strong>Accuracy:</strong> {:.2f}</p>
                    <p style="color: #78290F;"><strong>AUC:</strong> {:.2f}</p>
                    <p style="color: #78290F;"><strong>Precision:</strong> {:.2f}</p>
                    <p style="color: #78290F;"><strong>Recall:</strong> {:.2f}</p>
                    <p style="color: #78290F;"><strong>F1 Score:</strong> {:.2f}</p>
                </div>
                <hr style="border: 1px solid #ff7d00;">
                """.format(
                    metrics['Accuracy'], 
                    metrics['AUC'], 
                    metrics['Precision'], 
                    metrics['Recall'], 
                    metrics['F1 Score']
                ), unsafe_allow_html=True
            )

            # Plot and display the AUC curve
            st.subheader("AUC Curve")
            plot_auc(y_true, y_proba)
            st.markdown("<hr style='border: 1px solid #ff7d00;'>", unsafe_allow_html=True)

        else:
            # Warning message with custom styling
            st.markdown(
                """
                <div style="display: flex; justify-content: center; align-items: center; height: 200px;">
                    <div style="text-align: center; background-color: #FFDDC1; padding: 20px; border-radius: 10px; border: 2px solid #FF7D00;">
                        <h3 style="color: #78290F;">⚠️ Please upload a CSV file to proceed.</h3>
                    </div>
                </div>
                <hr style="border: 1px solid #ff7d00;">
                """, unsafe_allow_html=True
            )
            st.markdown(
                """
                <div style="background-color: #ffecd1; color: #001524; padding: 20px; border-radius: 10px; font-family: 'Arial', sans-serif;">
                    <h2 style="text-align: center; color: #ff7d00;">🔍 Detailed Process Overview</h2>
                    <p style="font-size: 1.2em; text-align: justify; line-height: 1.6;">
                        The entire process was meticulously executed, leveraging insightful visualizations that guided us through the key aspects of <strong>Exploratory Data Analysis</strong>. A robust preprocessing strategy was employed to balance the classes of our response variable. However, despite these efforts, the pre-processing phase did not yield significant improvements.
                    </p>
                    <hr style="border: 1px solid #ff7d00;">
                    <p style="font-size: 1.2em; text-align: justify; line-height: 1.6;">
                        During the <strong>Modeling Phase</strong>, a <strong>Logistic Regression</strong> model was trained on the Training set and evaluated on the Out-of-Time set. The results indicated that the model prioritized discrimination capability over accuracy. Given our primary objective of minimizing false negatives—critical in credit risk modeling to prevent the severe consequences of misclassifying a "Not-default" client—the trade-off was deemed necessary.
                    </p>
                    <hr style="border: 1px solid #ff7d00;">
                    <p style="font-size: 1.2em; text-align: justify; line-height: 1.6;">
                        Subsequent model refinement using <strong>PyCaret</strong> was attempted, but it resulted in only marginal improvements in accuracy.
                    </p>
                    <hr style="border: 1px solid #ff7d00;">
                    <p style="font-size: 1.2em; text-align: justify; line-height: 1.6;">
                        I am still in the process of learning how to perform effective credit risk modeling and welcome any tips or suggestions on how I can improve my approach.
                    </p>
                    <
                    <a id="conclusion"></a>
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
            <hr style="border: 1px solid #ff7d00;">
            <p style="text-align: center; font-size: 22px;">
                This project is multifaceted, drawing on a diverse range of tools and techniques that are 
                <b>CRUCIAL</b> for ensuring thorough analysis and clear communication. These resources are integral to 
                the project's success, providing the foundation for robust and insightful results. Key resources include:
            </p>
            <hr style="border: 1px solid #ff7d00;">
            """, unsafe_allow_html=True
        )

        # Create side-by-side buttons with enhanced styling
        st.markdown(
            """
            <div style="display: flex; justify-content: space-around; margin-top: 20px;">
                <a href="https://youtube.com/your-video-link" target="_blank" style="background-color: #cc0000; color: white; padding: 16px 32px; text-decoration: none; border-radius: 8px; font-weight: bold; display: inline-block; width: 20%; text-align: center;">YouTube Video</a>
                <a href="https://kaggle.com/your-project-link" target="_blank" style="background-color: #0000cc; color: white; padding: 16px 32px; text-decoration: none; border-radius: 8px; font-weight: bold; display: inline-block; width: 20%; text-align: center;">Kaggle Project</a>
                <a href="https://github.com/PedroHang/Credit-Scoring-Project" target="_blank" style="background-color: #660066; color: white; padding: 16px 32px; text-decoration: none; border-radius: 8px; font-weight: bold; display: inline-block; width: 20%; text-align: center;">GitHub Repository</a>
                <a href="https://app.powerbi.com/view?r=eyJrIjoiM2ZjNzZkNTAtN2M5NS00MWJkLThjMDItMWFiOTg5NDJkYTgzIiwidCI6IjcxMWE5Mzc5LTI0MTMtNGYxMy04NTlmLTlhYzhkYzc2MjRhMyJ9" target="_blank" style="background-color: #cc6600; color: white; padding: 16px 32px; text-decoration: none; border-radius: 8px; font-weight: bold; display: inline-block; width: 20%; text-align: center;">PowerBI Dashboard</a>
            </div>
            <hr style="border: 1px solid #ff7d00;">
            """, unsafe_allow_html=True
        )

        st.markdown(
            """
            <p style="text-align: center; margin-top: 40px;">Check out my LinkedIn page:</p>
            <div style="display: flex; justify-content: center;">
                <a href="https://linkedin.com/in/your-profile-link" target="_blank" style="background-color: #0077cc; color: white; padding: 16px 32px; text-decoration: none; border-radius: 8px; font-weight: bold; display: inline-block; text-align: center;">LinkedIn Page</a>
            </div>
            <hr style="border: 1px solid #ff7d00;">
            """, unsafe_allow_html=True
        )

    elif page == "Dashboard":
        # Dashboard page content
        st.markdown(
            """
            <h1 style='text-align: center;'>
                <img src="https://cdn-icons-png.flaticon.com/512/1604/1604695.png" alt="icon" width="50"/>
                Dashboard
            </h1>
            <hr style="border: 1px solid #ff7d00;">
            <p style="text-align: center; font-size: 22px;">
                Below is the embedded Power BI dashboard for detailed insights.
            </p>
            <hr style="border: 1px solid #ff7d00;">
            """, unsafe_allow_html=True
        )

        # Embed the Power BI dashboard using an iframe with a wider width
        st.markdown(
            """
            <div style="display: flex; justify-content: center; margin-top: 20px;">
                <iframe width="100%" height="900" src="https://app.powerbi.com/view?r=eyJrIjoiM2ZjNzZkNTAtN2M5NS00MWJkLThjMDItMWFiOTg5NDJkYTgzIiwidCI6IjcxMWE5Mzc5LTI0MTMtNGYxMy04NTlmLTlhYzhkYzc2MjRhMyJ9" frameborder="0" allowFullScreen="true"></iframe>
            </div>
            <hr style="border: 1px solid #ff7d00;">
            """, unsafe_allow_html=True
        )

        # Provide a clickable link for the dashboard
        st.markdown(
            """
            <div style="text-align: center; margin-top: 20px;">
                <a href="https://app.powerbi.com/view?r=eyJrIjoiM2ZjNzZkNTAtN2M5NS00MWJkLThjMDItMWFiOTg5NDJkYTgzIiwidCI6IjcxMWE5Mzc5LTI0MTMtNGYxMy04NTlmLTlhYzhkYzc2MjRhMyJ9" target="_blank" style="background-color: #ff7d00; color: white; padding: 10px 20px; text-decoration: none; border-radius: 8px; font-weight: bold; display: inline-block;">Power BI Dashboard</a>
            </div>
            """, unsafe_allow_html=True
        )

# Run the app
if __name__ == "__main__":
    main()
