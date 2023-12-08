import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('train.csv')

# Encode the categorical features
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

# Create a boolean mask for filtering out missing values
mask = df['Age'].isnull() & df['Embarked'].isnull()

# Create a new feature 'Cabin' indicating whether or not a passenger had a cabin
#df['Cabin'] = df['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)

# Use the 'Fare' feature to replace the missing 'Age' and 'Embarked' values
df.loc[mask, 'Age'] = df.loc[mask, 'Fare']
df.loc[mask, 'Embarked'] = 0

# Create a binary target variable 'Survived' indicating whether or not a passenger survived
df['Survived'] = df['Survived'].map({0: 0, 1: 1})

# Select the features for training the model
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features]
y = df['Survived']

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rfc.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rfc.predict(X_test)

# Print the accuracy of the model
print("Accuracy: ", accuracy_score(y_test, y_pred))