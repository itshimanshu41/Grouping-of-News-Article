## Grouping of News Article

### Overview
This project groups/classifies news articles as REAL or FAKE using classic NLP preprocessing and a Logistic Regression classifier trained on TF‑IDF features. The workflow lives in the notebook `Grouping_of_News_Article.ipynb` and uses the dataset `Grouping of News.csv` in the repository root.

### Dataset
- **File**: `Grouping of News.csv`
- **Columns**:
  - `title`: Headline of the article
  - `text`: Full article content
  - `label`: Target class (`REAL` or `FAKE`)
  - An index/id column may be present as the first column when loaded; it is not used by the model

The notebook creates an auxiliary column:
- `content = title + " " + text`

### Approach
1. **Text normalization**
   - Keep alphabetic characters only (regex `[^a-zA-Z]` → space)
   - Lowercasing
   - Tokenization via `split()`
   - Stopword removal using NLTK English stopwords
   - Lemmatization using NLTK `WordNetLemmatizer`
2. **Vectorization**
   - `TfidfVectorizer` to convert `content` to sparse features
3. **Split**
   - `train_test_split` with `test_size=0.2`, `stratify=Y`, `random_state=83`
4. **Model**
   - `LogisticRegression` (default settings in scikit‑learn)
5. **Evaluation**
   - Accuracy on train and test sets
   - Confusion matrix visualization (note: `plot_confusion_matrix` is deprecated; consider `ConfusionMatrixDisplay.from_estimator`)

### Results (from notebook)
- Train accuracy: ~95.501%
- Test accuracy: ~91.318%

These results indicate good generalization for a simple linear classifier on TF‑IDF features.

### Requirements
- Python 3.8+
- Packages:
  - `numpy`
  - `pandas`
  - `nltk`
  - `scikit-learn`
  - `matplotlib`

NLTK data required:
- `stopwords`
- `wordnet`

The notebook currently calls:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

### Setup
Install dependencies (one option):
```bash
pip install numpy pandas nltk scikit-learn matplotlib
```
Optionally pre‑download NLTK data to avoid interactive prompts:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

### Running the Notebook
1. Open `Grouping_of_News_Article.ipynb` in Jupyter or VS Code.
2. Ensure the CSV path is correct. The original notebook used a Google Colab path:
   ```python
   data = pd.read_csv('/content/drive/MyDrive/ML Dataset/Grouping of News Article.csv')
   ```
   Update it to the repository file path, for example:
   ```python
   data = pd.read_csv('Grouping of News.csv')
   ```
3. Run cells top‑to‑bottom.

### File Structure
- `Grouping of News.csv` — dataset with `title`, `text`, `label`
- `Grouping_of_News_Article.ipynb` — end‑to‑end workflow: preprocessing, TF‑IDF, train/test split, Logistic Regression, evaluation
- `README.md` — this documentation

### Notes and Recommendations
- Replacing `plot_confusion_matrix` with:
  ```python
  from sklearn.metrics import ConfusionMatrixDisplay
  ConfusionMatrixDisplay.from_estimator(model, X_test, Y_test)
  ```
- Consider a pipeline for reproducibility and cleaner code:
  ```python
  from sklearn.pipeline import make_pipeline
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.linear_model import LogisticRegression

  pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression())
  pipeline.fit(train_texts, train_labels)
  ```
- For stronger baselines, try `LinearSVC`, n‑grams in TF‑IDF, or class‑weight tuning if labels are imbalanced.

### License
If you intend to open‑source, add a license (e.g., MIT) to clarify terms of use.
