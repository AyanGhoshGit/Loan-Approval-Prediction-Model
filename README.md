# **Loan Approval Prediction Using Machine Learning**

In this project, I developed a **Loan Approval Prediction** system that uses machine learning techniques to determine whether a loan application should be approved or not, based on applicants’ financial and personal data. This project applies robust data preprocessing, outlier handling, and feature engineering techniques, followed by model training and evaluation using **Support Vector Classifier (SVC)** for accurate loan approval classification.

---

## **Overview**

Loan approval prediction involves analyzing multiple attributes of loan applicants such as income, employment status, credit history, and education level to assess the likelihood of loan approval. By training a supervised learning model on historical loan data, we can automate the decision-making process in financial institutions, improving accuracy and efficiency in loan disbursement.

---

## **Technologies Used**

* **Python** —> Primary programming language for data analysis and modeling.
* **Pandas / NumPy** —> Data manipulation and numerical computation.
* **Scikit-learn** —> Machine learning model implementation and preprocessing tools.
* **Plotly Express** —> Interactive visualizations for Exploratory Data Analysis (EDA).
* **StandardScaler** —> Feature scaling to standardize numerical data.

---

## **Dataset**

The dataset used for this project is `loan_prediction.csv`, containing information about loan applicants and their loan approval status.

### **Key Columns:**

* **Loan_ID:** Unique identifier for each loan application.
* **Gender:** Applicant’s gender (Male/Female).
* **Married:** Marital status of the applicant (Yes/No).
* **Dependents:** Number of dependents.
* **Education:** Education level (Graduate/Not Graduate).
* **Self_Employed:** Employment type (Yes/No).
* **ApplicantIncome:** Applicant’s income.
* **CoapplicantIncome:** Income of co-applicant (if any).
* **LoanAmount:** Requested loan amount.
* **Loan_Amount_Term:** Term/duration of the loan in months.
* **Credit_History:** Binary indicator showing creditworthiness (1 = Good, 0 = Poor).
* **Property_Area:** Property location (Urban/Rural/Semiurban).
* **Loan_Status:** Target variable — (Y = Approved, N = Not Approved).

---

## **Project Workflow**

### **1. Data Preprocessing**

* Loaded and explored `loan_prediction.csv`.
* Dropped non-informative columns (`Loan_ID`).
* Handled missing values:

  * **Categorical columns:** Filled with mode.
  * **Numerical columns:**

    * `LoanAmount` → filled with median.
    * `Loan_Amount_Term` & `Credit_History` → filled with mode.
* Converted categorical variables into numeric format using **One-Hot Encoding**.

### **2. Exploratory Data Analysis (EDA)**

* Visualized loan approval distributions and demographic factors using **Plotly Express**:

  * Loan Status distribution (Approved vs. Not Approved).
  * Gender, Marital Status, Education, and Employment distributions.
  * Relationship analysis:

    * Loan_Status vs ApplicantIncome
    * Loan_Status vs CoapplicantIncome
    * Loan_Status vs LoanAmount
    * Loan_Status vs Credit_History
    * Loan_Status vs Property_Area
* Removed outliers from **ApplicantIncome** and **CoapplicantIncome** using **Interquartile Range (IQR)** method.

### **3. Feature Engineering**

* Applied **One-Hot Encoding** to categorical columns:

  * `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `Property_Area`.
* Scaled numerical columns using **StandardScaler**:

  * `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`.

### **4. Model Training**

* Split data into training (80%) and testing (20%) sets.
* Trained an **SVM Classifier (SVC)** to predict loan approval status.
* Evaluated model accuracy on the test set.

### **5. Prediction and Evaluation**

* Generated predictions on test data (`y_pred`).
* Combined predicted values with test data for comparison.
* Evaluated model performance using classification metrics (accuracy, precision, recall, F1-score — optional future extension).

---

## **Run the Project**

1. **Install Dependencies:**

   ```bash
   pip install pandas numpy scikit-learn plotly
   ```

2. **Run the Python Script:**

   ```bash
   python loan_approval_prediction.py
   ```

3. **View Visualizations:**
   Interactive EDA plots will open in your browser using Plotly Express.

---

## **Model Output**

* The model predicts whether a loan will be **approved ("Y")** or **not approved ("N")** based on applicant attributes.
* Final output includes:

  * Preprocessed features.
  * Scaled numerical data.
  * Predicted loan status appended as a new column `Loan_Status_Predicted`.


