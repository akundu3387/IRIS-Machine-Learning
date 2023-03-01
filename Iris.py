import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Scale the feature data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.transform(X_test)

# Train and evaluate the linear regression model
from sklearn.linear_model import LinearRegression 
linear_reg = LinearRegression()
linear_reg.fit(rescaledX_train, y_train)
y_pred_lr = linear_reg.predict(rescaledX_test)
print("Linear Regression has score: ", linear_reg.score(rescaledX_test, y_test))
print("Linear Regression has weights: ",linear_reg.coef_)
w = np.linalg.inv(rescaledX_train.T.dot(rescaledX_train)).dot(rescaledX_train.T).dot(y_train)
print("Analytical weights: ", w)

# Train and evaluate the Perceptron model
from sklearn.linear_model import Perceptron
percep = Perceptron(max_iter=1000, tol=1e-3)
percep.fit(rescaledX_train,y_train)
y_pred_pc = percep.predict(rescaledX_test)
print("\nPerceptron has a score: ",percep.score(rescaledX_test,y_test))
print("Perceptron has weights: ",percep.coef_)
print("Perceptron confusion matrix:\n", confusion_matrix(y_test, y_pred_pc))

# Train and evaluate the Logistic regression Model
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(rescaledX_train, y_train)
y_pred_log = log_reg.predict(rescaledX_test)
print("\nLogistic Regression has a score: ", log_reg.score(rescaledX_test, y_test))
print("Logistic Regression has weights: ", log_reg.coef_)
print("Logistic Regression confusion matrix:\n", confusion_matrix(y_test, y_pred_log))
