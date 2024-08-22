# Credit Score Project

**Analyzing and Predicting Credit Defaults**

**Author:** Pedro Hang

## View the notebook on <a href="https://nbviewer.org/github/PedroHang/Credit-Scoring-Project/blob/main/credit_scoring_original.ipynb" target="_blank" style="background-color: #0000cc; color: white; padding: 12px 24px; text-decoration: none; border-radius: 8px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2); font-weight: bold;">NBviewer</a>
## <img src="https://cdn-icons-png.flaticon.com/512/174/174883.png" width="16" height="16"> [Watch the video](https://www.youtube.com/watch?v=6rbWRaVanaM)

## Resources

This project is multifaceted, drawing on a diverse range of tools and techniques that are **CRUCIAL** for ensuring thorough analysis and clear communication. These resources are integral to the project's success, providing the foundation for robust and insightful results. Key resources include:
<table>
    <tr>
        <td>
            <a href="https://www.youtube.com/watch?v=6rbWRaVanaM" target="_blank" style="background-color: #cc0000; color: white; padding: 12px 24px; text-decoration: none; border-radius: 8px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2); font-weight: bold;">YouTube Video</a>
        </td>
        <td>
            <a href="https://nbviewer.org/github/PedroHang/Credit-Scoring-Project/blob/main/credit_scoring_original.ipynb" target="_blank" style="background-color: #0000cc; color: white; padding: 12px 24px; text-decoration: none; border-radius: 8px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2); font-weight: bold;">NBviewer</a>
        </td>
        <td>
            <a href="https://github.com/PedroHang/Credit-Scoring-Project" target="_blank" style="background-color: #660066; color: white; padding: 12px 24px; text-decoration: none; border-radius: 8px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2); font-weight: bold;">GitHub Repository</a>
        </td>
        <td>
            <a href="https://app.powerbi.com/view?r=eyJrIjoiM2ZjNzZkNTAtN2M5NS00MWJkLThjMDItMWFiOTg5NDJkYTgzIiwidCI6IjcxMWE5Mzc5LTI0MTMtNGYxMy04NTlmLTlhYzhkYzc2MjRhMyJ9" target="_blank" style="background-color: #cc6600; color: white; padding: 12px 24px; text-decoration: none; border-radius: 8px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2); font-weight: bold;">PowerBI Dashboard</a>
        </td>
        <td>
            <a href="https://credit-scoring-project.onrender.com" target="_blank" style="background-color: #cc0000; color: white; padding: 12px 24px; text-decoration: none; border-radius: 8px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2); font-weight: bold;">Styled Web App</a>
        </td>
    </tr>
</table>

[![YOUTUBE VIDEO](https://img.youtube.com/vi/6rbWRaVanaM/0.jpg)](https://www.youtube.com/watch?v=6rbWRaVanaM)

---

## Objective

The objective of this project is to conduct Exploratory Data Analysis (EDA), Data Pre-processing, and Modeling to develop an algorithm that accurately predicts whether a client will default based on various features.

This project aims to practice and apply a broad range of techniques and tools that I have acquired over the past few months, with a particular focus on becoming more familiar with the credit risk modeling domain.

## Techniques Used

This modeling task presented significant challenges, particularly due to the imbalance in the dataset. A variety of techniques were employed to make the modeling feasible, including:

- **Univariate and Bivariate Analysis:** To understand the distribution and relationships between features.
- **Data Pre-processing:** Addressing missing values, removing outliers, and applying transformations like Principal Component Analysis (PCA) to reduce dimensionality.
- **SMOTE (Synthetic Minority Over-sampling Technique):** Used to balance the dataset by oversampling the underrepresented class, making the model more sensitive to default cases.
- **Scikit-learn Pipeline:** A custom data pre-processing pipeline was built using Scikit-learn to streamline the transformation process and ensure consistency across the modeling stages.
- **PyCaret:** Employed to identify the best-performing models based on specific metrics, facilitating a streamlined model selection process.

## Quick Setup Instructions

### Step 1: Install Anaconda Navigator

If you haven't already, download and install [Anaconda Navigator](https://www.anaconda.com/products/distribution). Anaconda simplifies package management and deployment.

### Step 2: Create a New Environment

Open Anaconda Navigator and create a new environment for this project. This helps to keep dependencies organized and prevents conflicts with other projects.

1. Open Anaconda Navigator.
2. Click on the **Environments** tab.
3. Click the **Create** button.
4. Name your environment (e.g., `credit-score-env`).
5. Choose the Python version you need (e.g., Python 3.8).
6. Click **Create** to create the environment.

### Step 3: Install Project Dependencies

Activate your new environment and open a terminal. Run the following command to install all required dependencies:

```bash
pip install pandas matplotlib seaborn numpy streamlit scikit-learn plotly boto3 scipy pycaret
```

### Step 4: Run the Project

After installing the dependencies, you are ready to run the project. 

#### Running the Streamlit File

1. Ensure you are in the correct directory where the main Streamlit file is located.
2. Open a terminal.
3. Activate your environment if it is not already activated:

```bash
conda activate credit-score-env
```

4. Run the Streamlit file using the following command:

```bash
streamlit run credit_score_app.py
```

5. This will start the Streamlit server, and you will see output in the terminal indicating that the app is running. You can view the app in your web browser by navigating to the URL provided, typically `http://localhost:8501`.

## Disclaimers

- The original dataset was available in Portuguese, and I have translated it into English for this project.
- I am still learning, and any advice or suggestions for improvement are welcome.

---

Thank you for your time! Check out my [LinkedIn page](https://www.linkedin.com/in/pedrohang/) for more information and updates on my work.
