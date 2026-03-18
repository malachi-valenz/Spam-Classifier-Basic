import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Function to remove punctuation and convert text to lowercase
def clean_text(text):
    return ''.join([char for char in text if char not in string.punctuation]).lower()

# Load dataset and keep only relevant columns (label and message)
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1','v2']]

# Rename columns for clarity
df.columns = ['label','text']

# Preprocess the text data using the cleaning function
df['clean_text'] = df['text'].apply(clean_text)

# Convert text into a matrix of token counts (Bag of Words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict labels on the test set
y_pred = model.predict(X_test)

# Display performance metrics (precision, recall, f1-score)
print(classification_report(y_test, y_pred))