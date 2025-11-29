# Emotion Detection

This project demonstrates emotion detection from text using NLP and machine learning techniques. It uses a labeled dataset (`Emotion_Dataset1.csv`) and a Jupyter notebook (`Emotion_dedection.ipynb`) for data preprocessing, feature engineering, and classification.

---

## 🚀 Features
- Tokenization, POS tagging, stemming, and lemmatization
- Frequency analysis of tokens, bigrams, trigrams, and n-grams
- Data cleaning and preprocessing with spaCy
- Feature extraction using TF-IDF
- Handling class imbalance with SMOTE
- Emotion classification using Random Forest
- Evaluation with classification report and confusion matrix heatmap

---

## 🛠️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Muhammad-Muzammil-Shah/Emotion-Detection.git
   cd Emotion-Detection
   ```
2. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # Or
   source venv/bin/activate  # macOS/Linux
   ```
3. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn nltk spacy seaborn imbalanced-learn matplotlib
   python -m spacy download en_core_web_sm
   ```

---

## 💡 Usage
1. Open `Emotion_dedection.ipynb` in Jupyter or Colab.
2. Run the notebook step-by-step to:
   - Preprocess and tokenize text data
   - Engineer features and extract n-grams
   - Train and evaluate emotion classifier
   - Visualize results with confusion matrix
3. Review the interpretation and key findings at the end of the notebook.

---

## 📦 Technologies
- Python
- Jupyter Notebook
- pandas, numpy, scikit-learn, nltk, spaCy, seaborn, imbalanced-learn, matplotlib

---

## 📃 License
Open source. Feel free to use, modify, and distribute.
