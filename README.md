# Documentation for News Dataset Processing and Analysis

## Overview
This code processes and analyzes a news dataset, with the goal of detecting fake and real news based on various features like entity extraction, sentiment analysis, article length, and popularity. The process involves data cleaning, feature extraction, and statistical analysis. Additionally, it generates visualizations and performs hypothesis testing to compare the popularity of fake vs. real news.

---

### **Imports**
```python
import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
import spacy
import numpy as np
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from scipy.stats import ttest_ind
```
- **pandas**: Data manipulation and analysis.
- **re**: Regular expressions for text cleaning.
- **BeautifulSoup**: Parsing HTML to remove tags.
- **nltk**: Natural Language Toolkit for stopword removal.
- **spacy**: NLP library for tokenization and entity extraction.
- **numpy**: Numerical operations.
- **matplotlib.pyplot**: Visualization for plots.
- **seaborn**: High-level interface for statistical graphics.
- **textblob**: For sentiment analysis (polarity and subjectivity).
- **scipy.stats**: For performing t-tests.

---

### **Data Cleaning and Preprocessing**

#### 1. **Stopwords Download**
```python
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
```
- **nltk.download('stopwords')**: Downloads a set of English stopwords.
- **stop_words**: A set containing common words to be excluded from text during processing.

#### 2. **Text Cleaning Functions**
```python
def clean_text(text):
    if isinstance(text, str): 
        if '<' in text and '>' in text:
            text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
    return text
```
- **clean_text**: Cleans HTML tags, removes special characters, and extra spaces from the text.

#### 3. **Normalization Function**
```python
def normalize_text(text):
    return text.lower() if isinstance(text, str) else text
```
- **normalize_text**: Converts the text to lowercase.

#### 4. **Tokenization and Stopwords Removal**
```python
def tokenize_and_remove_stopwords(text):
    if isinstance(text, str):
        doc = nlp(text)
        tokens = [token.text for token in doc if not token.is_stop and token.is_alpha]
        return " ".join(tokens)
    return text
```
- **tokenize_and_remove_stopwords**: Tokenizes text using SpaCy, removing stopwords and non-alphabetic tokens.

---

### **Data Loading and Transformation**

```python
df_fake = pd.read_csv('politifact_fake.csv')
df_real = pd.read_csv('politifact_real.csv')
```
- **df_fake, df_real**: Loads the fake and real news CSV files.

```python
df_real['title'] = df_real['title'].apply(lambda x: tokenize_and_remove_stopwords(normalize_text(clean_text(x))))
df_fake['title'] = df_fake['title'].apply(lambda x: tokenize_and_remove_stopwords(normalize_text(clean_text(x))))
```
- **Data Cleaning**: Applies text cleaning, normalization, and stopword removal to the 'title' columns of both datasets.

```python
df_real['label'] = 0
df_fake['label'] = 1
```
- **Label Assignment**: Adds labels to the datasets (`0` for real news, `1` for fake news).

```python
df_real['tweet_ids'] = df_real['tweet_ids'].astype(str)
df_fake['tweet_ids'] = df_fake['tweet_ids'].astype(str)
```
- **Type Conversion**: Converts `tweet_ids` to string to handle long numbers.

```python
df_combined = pd.concat([df_fake, df_real], ignore_index=True)
```
- **Data Concatenation**: Merges the fake and real news datasets into one combined dataset.

```python
df_combined.to_csv('combined_cleaned_data.csv', index=False)
```
- **Saving Data**: Saves the combined and cleaned dataset to a CSV file.

---

### **Entity Extraction with SpaCy**
```python
def extract_entities_spacy(text):
    if not isinstance(text, str) or text.strip() == "":
        return {"PERSON": 0, "ORG": 0, "GPE": 0}
    doc = nlp(text)
    entity_counts = {"PERSON": 0, "ORG": 0, "GPE": 0}
    for ent in doc.ents:
        if ent.label_ in entity_counts:
            entity_counts[ent.label_] += 1
    return entity_counts
```
- **extract_entities_spacy**: Extracts named entities (PERSON, ORG, GPE) from text using SpaCy's NER model.

```python
entity_data = df["title"].apply(extract_entities_spacy)
```
- **Entity Extraction**: Applies the entity extraction function to the 'title' column.

```python
entity_df = pd.DataFrame(entity_data.tolist())
df = pd.concat([df, entity_df], axis=1)
```
- **Merge Entity Data**: Converts the entity extraction results into a DataFrame and merges them with the main DataFrame.

---

### **Popularity Calculation**
```python
df["popularity"] = df["tweet_ids"].apply(lambda x: len(str(x).split(",")) if pd.notna(x) else 0)
```
- **Popularity**: Calculates popularity based on the number of tweet IDs (assuming comma-separated).

---

### **Sentiment Analysis**
```python
def calculate_sentiment(text):
    if not isinstance(text, str) or text.strip() == "":
        return {"polarity": 0, "subjectivity": 0}
    blob = TextBlob(text)
    return {"polarity": blob.sentiment.polarity, "subjectivity": blob.sentiment.subjectivity}
```
- **calculate_sentiment**: Analyzes the sentiment of the text using TextBlob, returning polarity and subjectivity.

```python
sentiment_data = df["title"].apply(calculate_sentiment)
sentiment_df = pd.DataFrame(sentiment_data.tolist())
df = pd.concat([df, sentiment_df], axis=1)
```
- **Sentiment Extraction**: Applies sentiment analysis and merges the results with the main DataFrame.

---

### **Article Length Calculation**
```python
def calculate_article_length(text):
    if not isinstance(text, str) or text.strip() == "":
        return 0
    return len(text.split())
```
- **calculate_article_length**: Computes the length of the article (word count).

```python
df["article_length"] = df["title"].apply(calculate_article_length)
```
- **Apply Article Length**: Adds the article length as a new column to the DataFrame.

---

### **Statistical Analysis and Visualization**

#### 1. **Correlation Heatmap**
```python
correlation = df[["popularity", "PERSON", "ORG", "GPE", "label"]].corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
```
- **Correlation**: Calculates the correlation matrix between popularity, entity counts, and labels. Then visualizes it using a heatmap.

#### 2. **T-test for Popularity Difference**
```python
t_stat, p_value = ttest_ind(fake_popularity, real_popularity, equal_var=False)
```
- **T-test**: Performs a t-test to compare the means of the popularity of fake and real news articles.

#### 3. **Boxplot for Popularity Distribution**
```python
sns.boxplot(data=df, x="label", y="popularity", palette="Set2")
plt.title("Popularity Distribution by News Type (Fake vs. Real)")
plt.show()
```
- **Boxplot**: Visualizes the distribution of popularity for fake vs. real news articles.

---
