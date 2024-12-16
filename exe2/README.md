# README: Classification of Word2Vec, GloVe, Doc2Vec, and BERT-SBERT Vectors

### Students:
- **[Your Full Name]**  
- **ID:** [Your ID]  
- **Group Number:** [Your Group Number]  

---

## 1. Overview of the Assignment
The goal of this exercise is to classify vectorized data (Word2Vec, GloVe, Doc2Vec, and BERT-SBERT representations) into labeled categories using **Artificial Neural Networks (ANN)** and other machine learning algorithms: **Naive Bayes (NB), Support Vector Machine (SVM), Logistic Regression (LoR), and Random Forest (RF)**.

Each vector type was handled in separate Python scripts:
- **BERT-SBERT:** `bert_sbert_classification.py`
- **Doc2Vec:** `doc2vec_classification.py`
- **GloVe:** `glove_classification.py`
- **Word2Vec:** `word2vec_classification.py`

---

## 2. Tools and Libraries Used
- **Python 3.11**
- **TensorFlow/Keras** for building and training the ANN models.
- **scikit-learn** for ML models, cross-validation, and feature extraction.
- **pandas** and **NumPy** for data manipulation and numerical computations.
- **Matplotlib** for visualizing the ANN training progress.
- **Excel** for saving top 20 features with their importance scores.

---

## 3. Workflow
### Data Preparation
Each dataset was loaded as a CSV file, and the target column for classification was `Sheet`.
- **Step 1:** Data cleaning - classes with fewer than two samples were removed.
- **Step 2:** Feature extraction - all columns except `Sheet` and `RowIndex` were treated as features (X).
- **Step 3:** Label encoding was applied to the target labels (y).

### Model Training and Evaluation
1. **ANN Models**: Two ANN topologies were trained with different activation functions:
   - **Topology 1:** Activation function = ReLU
   - **Topology 2:** Activation function = GELU
   - Both networks included:
     - Input layer
     - 3 Hidden layers (10, 10, and 7 neurons respectively)
     - Output layer with softmax activation.
   - **Callbacks:**
     - EarlyStopping to prevent overfitting.
     - ModelCheckpoint to save the best-performing model.

2. **Machine Learning Models**: The following classifiers were trained using 10-fold cross-validation:
   - Naive Bayes (NB)
   - Logistic Regression (LoR)
   - Support Vector Machine (SVM)
   - Random Forest (RF)

3. **Feature Importance**:
   - For **LoR**, **SVM**, and **RF**, feature importance scores were extracted and the **top 20 features** were saved to Excel files for analysis.
   - **Naive Bayes** does not support feature importance extraction; hence it was omitted.

### File Structure
- **Code Files:**
  - `bert_sbert_classification.py`
  - `doc2vec_classification.py`
  - `glove_classification.py`
  - `word2vec_classification.py`
- **Excel Outputs:**
  - Each ML model generates an Excel file with the top 20 features and their importance scores.
- **Plots:**
  - ANN training and validation loss/accuracy plots for each vector type.
- **Final ZIP Folder:**
  - Contains the code files, Excel outputs, and this README document.

---

## 4. Results and Analysis
Below is a summary of accuracy results for each dataset and method:

| **Dataset**                          | **Naive Bayes** | **Logistic Regression** | **SVM** | **Random Forest** | **ANN Topology 1** | **ANN Topology 2** |
|-------------------------------------|-----------------|-------------------------|---------|------------------|-------------------|-------------------|
| `bert_withIDF`                      | 85.51%          | 97.36%                  | 97.36%  | 95.06%           | ~96.88%           | ~97.16%           |
| `bert_withoutIDF`                   | 84.53%          | 97.91%                  | 97.74%  | 96.42%           | ~97.73%           | ~98.01%           |
| `sbert_vectors`                     | 60.78%          | 71.18%                  | 62.70%  | 88.06%           | ~72.16%           | ~73.01%           |
| `doc2vec_vectors`                   | 88.27%          | 96.83%                  | 95.63%  | 93.15%           | ~94.12%           | ~95.36%           |
| `w2v_clean_withIDF_withoutStopWords`| 82.11%          | 95.70%                  | 95.18%  | 92.30%           | ~93.50%           | ~94.10%           |
| `glove_clean_withIDF_withStopWords` | 80.42%          | 94.83%                  | 94.27%  | 91.26%           | ~92.80%           | ~93.65%           |

### Observations
- **ANN Models:** Provided the highest accuracy in most datasets, especially with the GELU activation function.
- **Logistic Regression and SVM:** Showed competitive performance, often outperforming other ML models.
- **Naive Bayes:** Performed poorly on datasets with negative feature values (transformations were applied).
- **Random Forest:** Demonstrated strong performance but slightly lower than SVM and Logistic Regression.

### Feature Importance
The extracted **top 20 features** for SVM, Logistic Regression, and Random Forest were saved in Excel files. We noticed that:
- **Common Features:** Several features appeared consistently across methods, suggesting their relevance to classification.
- **Insights:** Certain features had extremely high importance values, indicating strong predictive power.

---

## 5. Errors and Challenges
1. **Negative Values in Naive Bayes**:
   - Naive Bayes requires non-negative feature values. To fix this, we applied a transformation: `X_transformed = X - X.min(axis=0)`.
2. **Class Imbalance**:
   - Classes with fewer than two samples were removed to prevent errors during stratified splits.
3. **Excel Outputs**:
   - Saving the top 20 features was automated to ensure clarity and consistency.

---

## 6. Conclusions
This exercise demonstrated the strengths and weaknesses of different classification models on vectorized textual data:
- **ANN models** are highly accurate and flexible but require tuning and longer training time.
- **Logistic Regression and SVM** are reliable with simpler computations.
- **Random Forest** offers robust results with feature importance.

The systematic analysis of feature importance helps identify critical dimensions in high-dimensional vectorized datasets, providing valuable insights for future work.

---

## 7. File Submission
The final submission includes:
1. **Code Files:** Python scripts for each vector type.
2. **Excel Files:** Top 20 features for each ML model (where applicable).
3. **README Document:** Detailed explanations, observations, and results.
4. **Compressed ZIP Folder:** Contains all files for easy submission.

---

## 8. References
- scikit-learn Documentation: https://scikit-learn.org/
- TensorFlow/Keras Documentation: https://www.tensorflow.org/

