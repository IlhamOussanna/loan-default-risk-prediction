# Loan Default Risk Prediction Using Machine Learning

This project aims to predict loan default risk using Lending Club data (2007–2018) by applying statistical analysis and machine learning models like **Logistic Regression**, **Random Forest**, and **XGBoost**. It uses a full CRISP-DM pipeline including preprocessing, feature selection, class balancing (SMOTE), dimensionality reduction (PCA), and model interpretability (SHAP).

## Key Highlights

- **Dataset**: Lending Club (2007–2018) – 2.2M records, 150+ features  
- **Data Preprocessing**: Null removal, feature encoding, SMOTE balancing  
- **Statistical Analysis**: Z-tests for key financial variables  
- **Models Used**: Logistic Regression, Random Forest, XGBoost  
- **Feature Reduction**: PCA (retained 95% variance)  
- **Model Interpretability**: SHAP  
- **Best Performance**: **XGBoost** (ROC-AUC: 0.99, Recall: 0.90)

## Tools & Technologies

- Python (Pandas, NumPy, Scikit-learn, XGBoost)
- SHAP (Explainability)
- SMOTE (Class balancing)
- Jupyter Notebook
- Matplotlib & Seaborn

## Files

| File Name                         | Description                          |
|----------------------------------|--------------------------------------|
| `Lending_Club_Loan_Code.ipynb`   | Jupyter Notebook with full analysis  |
| `Loan_Default_Prediction_Report.pdf` | Project report & results         |
| `README.md`                      | Project summary & documentation      |

## Results

XGBoost outperformed other models with:

- **ROC-AUC**: 0.99  
- **Recall**: 0.90  
- **Precision**: 0.89  

The model demonstrated strong predictive performance and was interpreted using SHAP for transparency and fairness.

## Ethics & Fairness

This project emphasized ethical AI by auditing predictions for bias using SHAP. Sensitive attributes were monitored to avoid discrimination and ensure responsible use.

## Future Work

- Integrate macroeconomic indicators  
- Real-time API deployment with Flask/FastAPI  
- Regular retraining and fairness audits

## Dataset Reference

- [Lending Club Dataset on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
