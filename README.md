

# 🚢 Titanic Survival Prediction 🐍

This project uses a Random Forest Classifier to predict whether a passenger on the Titanic survived or not, based on features like age, class, sex, etc. The model is built using Python, Pandas, and Scikit-learn.

## 📝 Project Overview

This repository contains a Python script (`titanic_predictor.py` - *please rename appropriately if needed*) that performs the following steps:

1.  **Loads Data:** Reads the `Titanic-Dataset.csv` file (included in the repository).
2.  **Data Preprocessing:**
    *   **Drops Irrelevant Columns:** Removes `PassengerId`, `Name`, `Ticket`, and `Cabin`.
    *   **Handles Missing Values:**
        *   Fills missing `Age` values with the median age of all passengers.
        *   Fills missing `Embarked` values with the mode (most frequent port).
    *   **Converts Categorical Features:** Uses One-Hot Encoding (`pd.get_dummies`) to convert `Sex` and `Embarked` into numerical format, dropping the first category to avoid multicollinearity.
3.  **Feature/Target Split:** Separates the processed features (X) from the target variable (`Survived`).
4.  **Train-Test Split:** Divides the data into training (80%) and testing (20%) sets (`random_state=42` ensures reproducibility).
5.  **Model Training:** Initializes and trains a `RandomForestClassifier` (with 100 trees and `random_state=42`) on the training data.
6.  **Prediction:** Uses the trained model to predict survival outcomes (0 or 1) for the test set.
7.  **Evaluation:**
    *   Calculates and prints the overall **Accuracy** of the model on the test set.
    *   Prints a detailed **Classification Report**, showing precision, recall, and F1-score for both classes (Survived=0 and Survived=1).

## 💾 Dataset

*   **File:** `Titanic-Dataset.csv` (Included in this repository)
*   **Source:** The classic Titanic dataset, commonly used for introductory machine learning tasks (often sourced from Kaggle).
*   **Columns (Original Relevant):** `Survived`, `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`.
*   **Target:** `Survived` (0 = No, 1 = Yes).

## ✨ Features & Target

*   **Features (X) (after preprocessing):** `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`, `Sex_male`, `Embarked_Q`, `Embarked_S`.
*   **Target Variable (y):** `Survived`.

## ⚙️ Technologies & Libraries

*   Python 3.x
*   Pandas
*   NumPy
*   Scikit-learn (`train_test_split`, `RandomForestClassifier`, `accuracy_score`, `classification_report`)

## 🛠️ Setup & Installation (using VS Code)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/PrabuMS07/Titanic-Survival-Prediction.git
    cd Titanic-Survival-Prediction
    ```
2.  **Open Folder:** Open the cloned folder in Visual Studio Code (`File` > `Open Folder...`).
3.  **Python Interpreter:** Ensure you have a Python 3 interpreter selected in VS Code.
4.  **Terminal:** Open the integrated terminal in VS Code (`View` > `Terminal` or `Ctrl + \``).
5.  **(Optional but Recommended) Create Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows PowerShell: .\venv\Scripts\Activate.ps1 or cmd: venv\Scripts\activate.bat
    ```
6.  **Install Dependencies:** Create a `requirements.txt` file in the folder with this content:
    ```txt
    pandas
    numpy
    scikit-learn
    ```
    Then run the installation command in the terminal:
    ```bash
    pip install -r requirements.txt
    ```
7.  **Dataset:** The `Titanic-Dataset.csv` file is already included in the repository.

## ▶️ Usage (within VS Code)

1.  Make sure you have completed the Setup steps (and activated the virtual environment if created).
2.  Open your Python script file (e.g., `titanic_predictor.py` - **you'll need to save the script code you provided into a `.py` file**) in the VS Code editor.
3.  Run the script from the VS Code terminal:
    ```bash
    python your_script_name.py
    ```
    *(Replace `your_script_name.py` with the actual name of your Python file)*

4.  **Output:** The script will print the following directly into the VS Code Terminal:
    *   The calculated `Accuracy` score (formatted to two decimal places).
    *   The detailed `Classification Report` showing precision, recall, and F1-score for predicting both non-survival (0) and survival (1).

## 📊 Interpreting the Results

*   **Accuracy:** The overall percentage of passengers whose survival status was correctly predicted by the model on the test set.
*   **Classification Report:**
    *   **Precision:** For a given prediction (e.g., predicted 'Survived'), what proportion of those predictions were correct?
    *   **Recall:** For a given actual outcome (e.g., actual 'Survived'), what proportion did the model correctly identify?
    *   **F1-Score:** The harmonic mean of precision and recall, useful for balancing these metrics, especially in cases of class imbalance (though less pronounced in this dataset).

## 💡 Potential Future Improvements

*   **Feature Engineering:**
    *   Create new features like `FamilySize` (SibSp + Parch + 1).
    *   Extract titles from the 'Name' column (e.g., Mr., Mrs., Miss., Master.) before dropping it, as titles can correlate with age, sex, and status.
    *   Bin numerical features like `Age` or `Fare` into categories.
*   **Advanced Imputation:** Use more sophisticated methods for imputing `Age` (e.g., regression imputation based on other features) instead of simple median.
*   **Feature Scaling:** Apply `StandardScaler` or `MinMaxScaler` to numerical features (`Age`, `Fare`, `Pclass`, `SibSp`, `Parch`) before training. While Random Forest is less sensitive to scaling than some models, it can sometimes help.
*   **Hyperparameter Tuning:** Use `GridSearchCV` or `RandomizedSearchCV` to optimize the `RandomForestClassifier` parameters (`n_estimators`, `max_depth`, `min_samples_split`, etc.).
*   **Cross-Validation:** Use k-fold cross-validation for a more reliable estimate of the model's generalization performance.
*   **Explore Other Models:** Compare Random Forest performance with models like Logistic Regression, Support Vector Machines (SVM), Gradient Boosting (like XGBoost or LightGBM), etc.
*   **Add Visualizations:** Use `matplotlib` and `seaborn` to visualize data distributions, correlations, feature importances, or confusion matrices.

---
