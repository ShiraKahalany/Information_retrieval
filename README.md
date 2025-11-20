# 📚 Information Retrieval System

### 📌 Overview  
The Information Retrieval System is a robust tool designed to efficiently process and retrieve information from vast datasets. Utilizing advanced techniques in natural language processing and machine learning, this system addresses the challenge of effectively searching and analyzing textual data. It is tailored for researchers, data analysts, and developers who require precise and rapid access to relevant information. The code implements various algorithms, including TF-IDF and BERT, to enhance search functionality and relevance scoring.

### ✨ Features  
- **TF-IDF Analysis**: Computes Term Frequency-Inverse Document Frequency to rank the importance of words in documents.  
- **BERT Integration**: Leverages the BERT model for semantic search capabilities, ensuring more contextual results.  
- **Text Cleaning and Preprocessing**: Includes scripts for cleaning and stemming text data to improve processing accuracy.  
- **Vector Storage**: Efficient storage of vector representations from models for quick retrieval during searches.  
- **Data Management**: Organized management of cleaned text and matrix files for seamless data handling and processing.  

### 🛠 Tech Stack  
- **Languages**: Python  
- **Libraries**: NumPy, Pandas, Scikit-learn  
- **Tools**: BERT, TF-IDF  

### 🏗 Architecture  
The system follows a modular architecture, primarily focusing on data processing and retrieval functionalities, leading to enhanced scalability and maintainability. The design separates concerns between text preprocessing, algorithm implementation (like TF-IDF and BERT), and data storage. This modularity ensures that components can be updated or replaced independently without affecting overall system performance.

### 📂 Folder Structure  
```
├── exe1/
│   ├── Sentence-BERT_matrices/
│   │   ├── A-J_Sentence-BERT_vectors.txt
│   │   ├── BBC_Sentence-BERT_vectors.txt
│   │   ├── J-P_Sentence-BERT_vectors.txt
│   │   └── NY-T_Sentence-BERT_vectors.txt
│   ├── clean&stem posts.py
│   ├── TF-IDF.py
│   └── bert.py
└── clean_texts/
    ├── A-J/
    │   ├── A-J_0_clean.txt
    │   ├── A-J_100_clean.txt
    │   ├── A-J_101_clean.txt
    │   ├── A-J_102_clean.txt
    │   ├── A-J_103_clean.txt
    │   └── A-J_105_clean.txt
```
