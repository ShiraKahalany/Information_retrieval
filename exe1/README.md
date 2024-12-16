# README - Text Processing and Feature Importance Analysis

---

## **Student Information**
- **Names**: 
  - Hadar Sarusi - 213383110
  - Shira Kahalany - 325283026
- **Class Number**: 
  - 157125.21.5785.42

---

## **Introduction**

In this project, we analyzed and processed text documents from four groups: `A-J`, `BBC`, `J-P`, and `NY-T`. The main objectives were:
1. **Clean and Prepare Text**:
   - Remove unnecessary elements and standardize content.
   - Apply stemming to reduce words to their root forms.
2. **Generate Document Representations**:
   - Using **TF-IDF**, **Word2Vec**, **Doc2Vec**, **BERT**, and **Sentence-BERT**.
3. **Analyze Feature Importance**:
   - Compute feature importance for the TF-IDF matrices using:
     - **Information Gain**.
     - **Chi-squared statistic**.
4. **Summarize and Extract Insights**.

---

## **Workflow**

### **Step 1: Text Cleaning and Preparation**
- **Objective**: Prepare the raw text for analysis by standardizing its structure and removing irrelevant elements.
- **Process**:
  1. **Tokenization**: Split the text into individual words.
  2. **Punctuation Removal**: Strip out special characters like commas, periods, and quotes.
  3. **Stopword Removal**: Exclude commonly used words (e.g., "and", "the") that don't carry significant meaning.
  4. **Stemming**:
     - **Algorithm**: The **Porter Stemming Algorithm** was used to reduce words to their root form.
     - Example: 
       - `running` → `run`
       - `connectivity` → `connect`

### **Step 2: Vector Representations**
We created document vectors using various algorithms:

#### **1. TF-IDF**
- **Overview**: Calculates the importance of terms (words or stems) within documents based on their frequency and uniqueness across the corpus.
- **Implementation**:
  - Two versions:
    1. **Cleaned TF-IDF**: Based on the cleaned text.
    2. **Stemmed TF-IDF**: Based on the stemmed text.
  - **Output**: Stored in the `tf_idf_matrices` folder.

#### **2. Word2Vec**
- **Overview**: Produces dense vector representations for each word by training on the local dataset.
- **Algorithm**:
  - Continuous Bag of Words (CBOW): Predicts a word based on its surrounding context.
  - Skip-gram: Predicts surrounding words given a word.
- **Process**:
  - Calculate the average of word vectors to generate a document vector.
  - Saved in the `word2vec_matrices` folder.

#### **3. Doc2Vec**
- **Overview**: Extends Word2Vec to generate unique vectors for each document.
- **Algorithm**: Trains on full documents to capture their context.
- **Output**: Saved in the `doc2vec_matrices` folder.

#### **4. BERT**
- **Overview**: Leverages the BERT model for contextual embeddings, limited to 512 tokens per document.
- **Output**: Saved in the `bert_matrices` folder.

#### **5. Sentence-BERT**
- **Overview**: Uses Sentence-BERT to generate embeddings that capture sentence-level semantics.
- **Output**: Saved in the `Sentence-BERT_matrices` folder.

### **Step 3: Feature Importance Analysis**
#### **A. Metrics Used**:
1. **Information Gain**:
   - Measures the reduction in uncertainty (entropy) for document classification when a specific feature is included.
   - Calculated using `mutual_info_classif` from `scikit-learn`.

2. **Chi-squared Statistic**:
   - Measures dependency between a feature and a binary target variable (classification label).
   - Normalized the TF-IDF matrix using `MinMaxScaler` before applying the Chi-squared test.

#### **B. Label Creation**:
- Labels were dynamically generated based on the importance of documents:
  1. Compute the sum of all feature values for each document (row).
  2. Define a threshold as the median of these sums.
  3. Assign `1` to rows above the median and `0` to rows below.

#### **C. Output**:
- Each matrix (`Cleaned` and `Stemmed`) for all groups (`A-J`, `BBC`, `J-P`, `NY-T`) was analyzed, and the results were saved in an Excel file:
  - **File Name**: `tfidf_feature_importance.xlsx`.
  - Each sheet contains:
    - Feature Index.
    - Information Gain Score.
    - Chi-squared Statistic.

---

## **Results**

### **Example Table for `A-J (Cleaned)`**
| Feature | Information Gain | Chi-squared |
|---------|------------------|-------------|
| 0       | 0.123            | 4.56        |
| 1       | 0.234            | 5.67        |
| 2       | 0.345            | 3.45        |
| ...     | ...              | ...         |

### **Example Table for `BBC (Stemmed)`**
| Feature | Information Gain | Chi-squared |
|---------|------------------|-------------|
| 0       | 0.112            | 3.56        |
| 1       | 0.210            | 6.78        |
| 2       | 0.322            | 4.12        |
| ...     | ...              | ...         |

---

## **Challenges and Solutions**

1. **Dynamic Label Creation**:
   - Challenge: Mismatch between the number of rows in the matrix and the number of labels.
   - Solution: Dynamically generated labels based on the median row importance.

2. **Empty Rows in TF-IDF Matrices**:
   - Challenge: Some rows were entirely zero, causing issues in Chi-squared calculations.
   - Solution: Ensured all matrices were cleaned and validated before analysis.

3. **Error Saving Excel File**:
   - Challenge: `writer.save()` caused compatibility issues with `XlsxWriter`.
   - Solution: Replaced `save()` with `close()`.

---

## **Insights**

1. **Stemming vs. Non-Stemming**:
   - Stemming reduced dimensionality but slightly altered the interpretation of features.
   - Cleaned matrices preserved full words, making feature analysis more interpretable.

2. **Importance of Feature Selection**:
   - Information Gain emphasized features with higher variance and relevance.
   - Chi-squared highlighted statistically significant dependencies.

3. **Embedding Effectiveness**:
   - Word2Vec and Sentence-BERT embeddings captured semantic relationships beyond term frequency.

---

## **Generated Files and Folders**

1. **`tf_idf_matrices`**:
   - Cleaned and Stemmed TF-IDF matrices for all groups.

2. **`word2vec_matrices`**:
   - Document vectors based on Word2Vec.

3. **`doc2vec_matrices`**:
   - Document vectors based on Doc2Vec.

4. **`bert_matrices`**:
   - BERT embeddings for documents.

5. **`Sentence-BERT_matrices`**:
   - Sentence-BERT embeddings for documents.

6. **Excel Output**:
   - `tfidf_feature_importance.xlsx`: Includes feature importance for all groups.

---

