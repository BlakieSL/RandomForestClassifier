import numpy as np  # Import NumPy library
import matplotlib.pyplot as plt  # Import pyplot module from Matplotlib
from sklearn.datasets import make_moons  # Import make_moons function from scikit-learn
from sklearn.model_selection import train_test_split  # Import train_test_split function from scikit-learn
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier class from scikit-learn
from sklearn.metrics import accuracy_score  # Import accuracy_score function from scikit-learn

def plot_decision_boundary(clf, X, X_train, y_train, X_test, y_test, ax, title):
    # Create grid for decision boundary plot
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
        np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
    )# np.linspace creates evenly spaced values over a specified range, x[:, 0] means all rows from first column, .min() minimum value from feature 1

    # Predict for each grid point
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot data points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='.', label='Train')
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', label='Test')

    # Draw decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3)

    # Set plot settings
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(title)

# Step 1: Create dataset
X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)  # Generate dataset using make_moons function

# Step 2: Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split dataset into training and test sets

# Step 3: Use random forest as classifier
criteria = ['gini', 'entropy']  # Criteria for random forest
table_estimators = [5, 20, 100, 500]  # Different numbers of estimators in random forest

# Create subplots before iteration
fig, axes = plt.subplots(4, 2, figsize=(12, 16))  # Create subplots to display results for different parameter combinations

for i, criterion in enumerate(criteria):  # Loop through criteria for random forest
    for j, n_estimator in enumerate(table_estimators):  # Loop through number of estimators in random forest
        print(i, j)
        ax = axes[j, i]  # Current subplot
        clf = RandomForestClassifier(n_estimators=n_estimator, criterion=criterion, bootstrap=True, max_depth=5, random_state=42)  # Initialize random forest classifier
        clf.fit(X_train, y_train)  # Fit model to training data
        y_pred_train = clf.predict(X_train)  # Make predictions on training data
        y_pred_test = clf.predict(X_test)  # Make predictions on test data
        train_accuracy = accuracy_score(y_train, y_pred_train)  # Calculate training accuracy
        test_accuracy = accuracy_score(y_test, y_pred_test)  # Calculate test accuracy

        title = (f"Crit:{criterion}, n trees:{n_estimator}\n"
                 f"Train acc:{train_accuracy:.4f}, Test acc:{test_accuracy:.4f}")

        # Use the plotting function instead of repetitive code
        plot_decision_boundary(clf=clf,X=X, X_train=X_train,y_train=y_train,X_test=X_test, y_test=y_test,ax=ax,title=title)

plt.tight_layout()  # Adjust plots to fit area
plt.legend()  # Add legend
plt.savefig("01_random_forest.png")  # Save plot to file
plt.show()  # Display plot