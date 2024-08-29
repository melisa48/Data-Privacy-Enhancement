import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Larger and more balanced dataset
data = {
    'text': [
        "John Doe's credit card number is 1234-5678-9101-1121.",
        "Alice lives in New York City.",
        "Our office is located in downtown.",
        "The price of the stock increased by 5%.",
        "Maria's email is maria@example.com.",
        "The bank account number is 123456789.",
        "Temperature is expected to be 25°C tomorrow.",
        "Confidential: Employee ID 987654321.",
        "The quick brown fox jumps over the lazy dog.",
        "Today is a sunny day.",
        "Send the payment to account number 123456.",
        "Peter's passport number is ABC123456.",
        "The movie was fantastic!",
        "Security code: 4567",
        "Here is my phone number: 555-1234.",
        "Social Security Number: 111-22-3333.",
        "This is a random sentence.",
        "The meeting is scheduled for Monday.",
        "Please call me at 555-6789.",
        "His bank details are private.",
        "James Bond's email is bond007@mi6.co.uk.",
        "The weather is nice today.",
        "The project deadline is next week.",
        "The company is launching a new product.",
        "The event will be held in the main hall."
    ],
    'label': [
        "sensitive",  # Contains a credit card number
        "sensitive",  # Contains a person's name and city
        "non-sensitive",
        "non-sensitive",
        "sensitive",  # Contains email
        "sensitive",  # Contains bank account number
        "non-sensitive",
        "sensitive",  # Contains employee ID
        "non-sensitive",
        "non-sensitive",
        "sensitive",  # Contains account number
        "sensitive",  # Contains passport number
        "non-sensitive",
        "sensitive",  # Contains security code
        "sensitive",  # Contains phone number
        "sensitive",  # Contains SSN
        "non-sensitive",
        "non-sensitive",
        "sensitive",  # Contains phone number
        "sensitive",  # Contains bank details
        "sensitive",  # Contains email
        "non-sensitive",
        "non-sensitive",
        "non-sensitive",
        "non-sensitive"
    ]
}

# Load data into a DataFrame
df = pd.DataFrame(data)

# Initialize spaCy model
nlp = spacy.load("en_core_web_trf")

# Function to preprocess text data
def preprocess_text(text):
    doc = nlp(text)
    return " ".join(token.lemma_ for token in doc if not token.is_stop)

# Preprocess the text data
df['processed_text'] = df['text'].apply(preprocess_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed_text'])
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Train the best model
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Test the model with a new example
new_text = "Jane Smith's social security number is 987-65-4321."
new_processed_text = preprocess_text(new_text)
new_vector = vectorizer.transform([new_processed_text])
new_prediction = best_model.predict(new_vector)

# Print the NER entities
doc = nlp(new_text)
entities = [(ent.text, ent.label_) for ent in doc.ents]
print("Entities found:", entities)

# Print the classification result
print("Classification:", new_prediction[0])




