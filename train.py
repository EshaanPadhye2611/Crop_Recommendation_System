import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score
import matplotlib.pyplot as plt
import pickle

# Load the dataset
data = pd.read_csv('C:/Users/eshaan/OneDrive/Desktop/Crop_Recommedation_System/Crop_recommendation.csv')  # Update with your dataset path

# Features and labels
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

# Splitting the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Decision Tree model
model = DecisionTreeClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')

# Classification report
print(classification_report(y_test, y_pred))

# Plotting the accuracy and precision
plt.figure(figsize=(10, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.bar(['Accuracy'], [accuracy], color='green')
plt.ylim([0, 1])
plt.title('Model Accuracy')
plt.ylabel('Accuracy Score')

# Precision plot
plt.subplot(1, 2, 2)
plt.bar(['Precision'], [precision], color='blue')
plt.ylim([0, 1])
plt.title('Model Precision')
plt.ylabel('Precision Score')

plt.tight_layout()
plt.show()

# Save the model to a file
with open('crop_recommendation_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model saved as 'crop_recommendation_model.pkl'")
