"""
SENTIMENT ANALYSIS PROJECT WITH NLP
Simple Machine Learning Project for Text Classification
"""

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
print("=" * 70)
print("SENTIMENT ANALYSIS PROJECT WITH NLP - MACHINE LEARNING")
print("=" * 70)

# ============= STEP 1: CREATE SAMPLE DATASET =============
print("\n[STEP 1] Creating Sample Dataset...")

positive_reviews = [
    "This movie is amazing and I loved every minute of it",
    "The plot was engaging and the acting was excellent",
    "Masterpiece! A must-watch for everyone",
    "Absolutely fantastic performance by all actors",
    "Best movie I have seen in years, highly recommend",
    "Great cinematography and brilliant screenplay",
    "I was completely blown away by this film",
    "Outstanding work from the entire cast and crew",
    "A true gem that deserves all the praise",
    "Spectacular production with amazing direction"
]

negative_reviews = [
    "Worst film I have ever seen in my entire life",
    "Boring and predictable plot throughout the movie",
    "Terrible dialogue and extremely poor direction",
    "I fell asleep during this movie, it was so bad",
    "Waste of time and money, do not watch",
    "Poorly executed with bad acting and weak story",
    "Confusing plot with terrible cinematography",
    "One of the most disappointing films ever made",
    "Absolutely awful, could not sit through it",
    "Horrible acting and no engaging storyline"
]

# Create DataFrame
reviews = positive_reviews + negative_reviews
sentiments = ['positive']*len(positive_reviews) + ['negative']*len(negative_reviews)
df = pd.DataFrame({'review': reviews, 'sentiment': sentiments})

print(f" Dataset created with {len(df)} reviews")
print(f"  Positive: {(df['sentiment']=='positive').sum()}")
print(f"  Negative: {(df['sentiment']=='negative').sum()}")
print(f"\nSample Reviews:")
print(f"  Positive: '{df[df['sentiment']=='positive']['review'].iloc[0]}'")
print(f"  Negative: '{df[df['sentiment']=='negative']['review'].iloc[0]}'")

# ============= STEP 2: TEXT PREPROCESSING =============
print("\n[STEP 2] Text Preprocessing...")

def preprocess_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = text.lower()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords and non-alphanumeric characters
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)

# Apply preprocessing
df['cleaned_review'] = df['review'].apply(preprocess_text)

print("Text preprocessing completed")
print(f"\n  Original: {df['review'].iloc[0]}")
print(f"  Cleaned:  {df['cleaned_review'].iloc[0]}")

# ============= STEP 3: FEATURE EXTRACTION =============
print("\n[STEP 3] Feature Extraction (TF-IDF)...")

# Train-Test Split
X = df['cleaned_review']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Data split completed")
print(f"  Training samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"TF-IDF vectorization completed")
print(f"  Feature vector shape: {X_train_tfidf.shape}")
print(f"  Total features extracted: {X_train_tfidf.shape[1]}")

# ============= STEP 4: MODEL TRAINING =============
print("\n[STEP 4] Training Classification Models...")

models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
    'Linear SVM': LinearSVC(max_iter=2000, random_state=42)
}

results = {}

for model_name, model in models.items():
    print(f"\n  Training {model_name}...", end=' ')
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = {
        'model': model,
        'accuracy': accuracy,
        'predictions': y_pred
    }
    print(f"Accuracy: {accuracy:.4f}")

# ============= STEP 5: MODEL EVALUATION =============
print("\n[STEP 5] Detailed Model Evaluation...")

best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
best_accuracy = results[best_model_name]['accuracy']

print(f"\nBest Model: {best_model_name}")
print(f"  Best Accuracy: {best_accuracy:.4f}")

# Get predictions from best model
y_pred = best_model.predict(X_test_tfidf)

# Classification Report
print(f"\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred,labels=['negative','positive'], target_names=['Negative', 'Positive']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred,labels=['negative','positive'])
print(f"\nConfusion Matrix:")
print(f"                Predicted")
print(f"              Negative  Positive")
print(f"Actual Negative  {cm[0,0]:3d}       {cm[0,1]:3d}")
print(f"       Positive  {cm[1,0]:3d}       {cm[1,1]:3d}")

# ============= STEP 6: VISUALIZATION =============
print("\n[STEP 6] Creating Visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Sentiment Analysis Results', fontsize=16, fontweight='bold')

# Plot 1: Model Comparison
ax1 = axes[0, 0]
model_names = list(results.keys())
accuracies = [results[m]['accuracy'] for m in model_names]
colors = ['#2ecc71' if acc == max(accuracies) else '#3498db' for acc in accuracies]
bars = ax1.bar(model_names, accuracies, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Accuracy', fontweight='bold')
ax1.set_title('Model Accuracy Comparison', fontweight='bold')
ax1.set_ylim([0, 1.1])
ax1.grid(axis='y', alpha=0.3)
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    ax1.text(bar.get_x() + bar.get_width()/2, acc + 0.03, f'{acc:.3f}', 
             ha='center', va='bottom', fontweight='bold')

# Plot 2: Confusion Matrix
ax2 = axes[0, 1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, 
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            cbar_kws={'label': 'Count'})
ax2.set_ylabel('True Label', fontweight='bold')
ax2.set_xlabel('Predicted Label', fontweight='bold')
ax2.set_title(f'Confusion Matrix - {best_model_name}', fontweight='bold')

# Plot 3: Sentiment Distribution
ax3 = axes[1, 0]
sentiment_counts = df['sentiment'].value_counts()
wedges, texts, autotexts = ax3.pie(sentiment_counts.values, 
                                     labels=['Positive', 'Negative'],
                                     autopct='%1.1f%%',
                                     colors=['#2ecc71', '#e74c3c'],
                                     startangle=90,
                                     textprops={'fontweight': 'bold'})
ax3.set_title('Sentiment Distribution in Dataset', fontweight='bold')

# Plot 4: Review Length by Sentiment
ax4 = axes[1, 1]
df['review_length'] = df['review'].apply(len)
positive_lengths = df[df['sentiment'] == 'positive']['review_length']
negative_lengths = df[df['sentiment'] == 'negative']['review_length']
ax4.hist(positive_lengths, bins=8, alpha=0.7, label='Positive', color='#2ecc71', edgecolor='black')
ax4.hist(negative_lengths, bins=8, alpha=0.7, label='Negative', color='#e74c3c', edgecolor='black')
ax4.set_title('Review Length Distribution by Sentiment', fontweight='bold')
ax4.set_xlabel('Review Length (characters)', fontweight='bold')
ax4.set_ylabel('Frequency', fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('sentiment_analysis_results.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Visualizations saved as 'sentiment_analysis_results.png'")

# ============= STEP 7: PREDICTION FUNCTION =============
print("\n[STEP 7] Testing Sentiment Predictions...")

def predict_sentiment(review):
    """Predict sentiment of a new review"""
    cleaned = preprocess_text(review)
    features = tfidf.transform([cleaned])
    prediction = best_model.predict(features)[0]
    return prediction

# Test on new reviews
test_reviews = [
    "This movie was absolutely amazing and wonderful!",
    "Terrible film, I hated every second of it",
    "The cinematography was beautiful but plot was weak",
    "I loved it, best movie ever made!",
    "Waste of time, very disappointing experience"
]

print("\n--- SENTIMENT PREDICTIONS ON NEW REVIEWS ---\n")
for i, review in enumerate(test_reviews, 1):
    sentiment = predict_sentiment(review)
    print(f"{i}. Review: \"{review}\"")
    print(f"   Prediction: {sentiment.upper()} ✓\n")

# ============= SUMMARY =============
print("=" * 70)
print("PROJECT SUMMARY")
print("=" * 70)
print(f"""
✓ Total Reviews Analyzed: {len(df)}
✓ Best Model: {best_model_name}
✓ Best Accuracy: {best_accuracy:.2%}
✓ Features Extracted: {X_train_tfidf.shape[1]}
✓ Training Samples: {len(X_train)}
✓ Test Samples: {len(X_test)}

KEY TECHNIQUES USED:
1. Text Preprocessing (lowercase, tokenization, stopword removal, stemming)
2. Feature Extraction (TF-IDF vectorization)
3. Machine Learning Classification (Naive Bayes, Logistic Regression, SVM)
4. Model Evaluation (Accuracy, Confusion Matrix, Classification Report)
5. Visualization (Model comparison, confusion matrix, distributions)

OUTPUT FILES:
- sentiment_analysis_results.png: Visualization of all results
- sentiment_analysis.py: Complete project code

PROJECT COMPLETED SUCCESSFULLY! ✓
""")
print("=" * 70)
