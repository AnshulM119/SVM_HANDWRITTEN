# SVM_HANDWRITTEN
# Import required libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Load the digits dataset (built into scikit-learn)
digits = datasets.load_digits()

# --- THIS IS THE UPDATED PART ---
# Show the first 10 sample images in a single grid
# We create a 2x5 grid of subplots
fig, axes = plt.subplots(2, 5, figsize=(10, 5))

# Loop through the first 10 images (i=0 to 9) and their corresponding subplots
for i, ax in enumerate(axes.ravel()):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f"Digit: {digits.target[i]}")
    ax.axis('off') # Hide the x/y axis labels

plt.suptitle("Figure 1: Sample Digits from the Dataset", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for the suptitle
plt.show()
# --- END OF UPDATED PART ---


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.3, random_state=42
)

# Train an SVM modelgit init
model = SVC(kernel='rbf', gamma=0.001, C=1)
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

# Print accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# -------------------------------
# SHOW SOME PREDICTIONS (This is the same as before)
# -------------------------------
# Pick 9 random test images to display
random_indices = np.random.choice(len(X_test), 9, replace=False)

plt.figure(figsize=(8, 8))
for i, idx in enumerate(random_indices):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[idx].reshape(8, 8), cmap='gray')
    plt.title(f"Pred: {y_pred[idx]}\nTrue: {y_test[idx]}")
    plt.axis('off')

plt.suptitle("Figure 2: Random Predictions vs Actual Labels")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# -------------------------------
# CONFUSION MATRIX (This is the same as before)
# -------------------------------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=digits.target_names)
disp.plot(cmap='plasma')
plt.title("Figure 3: Confusion Matrix - SVM Digit Classifier")
plt.show()

