# Text Classification of Movie Reviews by Director Gender (Scikit-Learn)

This project applies **text classification techniques** to predict the **gender of movie directors** based on movie descriptions. Using Scikit-Learn models (Naive Bayes, Logistic Regression, and Linear SVM), we benchmark different approaches and evaluate their performance on an imbalanced dataset.

---

## Dataset

- **Source**: Extended dataset created from `movie_reviews_with_sentiment.pkl`.  
- **Size**: 166 rows, 11 columns  
- **Relevant Columns**:  
  - `movie_title`: Title of the movie  
  - `movie_info`: Movie description  
  - `director_gender`: Target variable (`male` or `female`)  
  - `sentiment`: Sentiment score (calculated previously with VADER)  
  - `movie_info_clean`: Preprocessed movie descriptions (lowercased, tokenized, stopwords removed, lemmatized).  

**Class Distribution**:  
- Male Directors: ≈ 81%  
- Female Directors: ≈ 19%  

This significant imbalance makes accuracy misleading and highlights the importance of F1 scores.

---

## Methodology

1. **Preprocessing**  
   - Cleaned and normalized movie descriptions using the same text preprocessing pipeline as in the earlier Text Preprocessing project.  

2. **Train/Test Split**  
   - 80% training, 20% testing  
   - Stratified by director gender to maintain distribution.  

3. **Models Evaluated**  
   - **Naive Bayes (MultinomialNB)**  
   - **Logistic Regression**  
   - **Linear SVM**  

4. **Hyperparameter Tuning**  
   - Performed **GridSearchCV** with 5-fold cross-validation.  
   - Optimized parameters such as n-gram ranges, minimum/maximum document frequency, and model-specific hyperparameters (e.g., `alpha`, `C`).  

5. **Evaluation Metrics**  
   - Accuracy  
   - Precision (macro)  
   - Recall (macro)  
   - F1-score (macro)  

---

## Results

### Final Comparison of Models

| Model               | Best Params                                                                 | CV F1-macro | Test Accuracy | Test Precision | Test Recall | Test F1-macro |
|---------------------|-----------------------------------------------------------------------------|-------------|---------------|----------------|-------------|---------------|
| **Naive Bayes**     | `alpha=0.1, ngram_range=(1,1), min_df=0.05, max_df=0.8`                    | 0.588       | 0.706         | 0.497          | 0.497       | 0.494         |
| **Logistic Regression** | `C=10, ngram_range=(1,2), min_df=0.05, max_df=0.8`                         | 0.582       | 0.647         | 0.460          | 0.460       | 0.460         |
| **Linear SVM**      | `C=10, ngram_range=(1,2), min_df=0.05, max_df=0.8`                         | 0.599       | 0.676         | 0.529          | 0.532       | 0.530         |

---

### Model Performance Comparison

Performance metrics across models (Accuracy, Precision, Recall, F1-score) are shown below:

<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/f0911b8a-db1c-455e-a2bf-ea359ebc6630" />


---

## Key Observations

- **Accuracy is misleading**: Models achieved ~65–70% accuracy, but macro F1-scores were much lower (~0.46–0.53).  
- This discrepancy is due to **class imbalance** (male directors ≈ 4x more frequent than female).  
- **Naive Bayes** achieved the highest accuracy but lower F1.  
- **Linear SVM** provided the best F1-score (≈0.53), making it the most reliable model for this imbalanced classification task.  
- **Logistic Regression** underperformed compared to the other two models.  

---

## Limitations & Future Work

- Dataset size is **small** and heavily **imbalanced**, limiting model performance.  
- To improve results:  
  - Use **oversampling (SMOTE)** or **undersampling** techniques.  
  - Try **ensemble models** (Random Forest, XGBoost).  
  - Explore **deep learning methods** with contextual embeddings (e.g., BERT).  

---

## Conclusion

- **Linear SVM** is the most effective classifier for this dataset, balancing precision and recall better than Naive Bayes and Logistic Regression.  
- This project highlights the importance of evaluating models with **F1-scores** when dealing with imbalanced datasets.  
