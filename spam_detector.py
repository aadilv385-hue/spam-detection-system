import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Dataset
data = {
    "message": [
        "Win free money now",
        "Claim your free prize",
        "Congratulations you won",
        "Get free recharge",
        "Win free iPhone now",
        "Claim lottery reward",
        "Exclusive offer just for you",
        "Free cash bonus",

        "Hello how are you",
        "Meeting at 5 pm",
        "Let's study together",
        "Project submission today",
        "Call me later",
        "Good morning friend",
        "Assignment completed"
    ],

    "label": [
        "spam",
        "spam",
        "spam",
        "spam",
        "spam",
        "spam",
        "spam",
        "spam",

        "ham",
        "ham",
        "ham",
        "ham",
        "ham",
        "ham",
        "ham"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Convert text to numbers
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(df["message"])

# Labels
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()

model.fit(X_train, y_train)

# User input
message = input("Enter message: ")

# Transform input
message_vector = vectorizer.transform([message])

# Prediction
prediction = model.predict(message_vector)

print("Prediction:", prediction[0])