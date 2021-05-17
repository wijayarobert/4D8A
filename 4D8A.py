import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

st.title("5D8A (5 Datasets 8 Algorithms)")
st.write("""
## Explore the best classifier algorithm for particular dataset.
""")

dataset_name = st.sidebar.selectbox("Select Dataset", ("Breast Cancer", "Iris", "Wine", "Digits"))

classifier_name = st.sidebar.selectbox("Select Classifier", ("K-Nearest Neighbors", 
                                                             "Random Forest", 
                                                             "Decision Tree", 
                                                             "Support Vector Machine",
                                                             "Gradient Boosting",
                                                             "AdaBoost",
                                                             "Gaussian Naive Bayes",
                                                             "Neural Network (MLP)"))

def get_dataset(dataset_name):
  if dataset_name == "Iris":
    data = datasets.load_iris()
  elif dataset_name == "Breast Cancer":
    data = datasets.load_breast_cancer()
  elif dataset_name == "Wine":
    data = datasets.load_wine()
  else:
    data = datasets.load_digits()
  X = data.data
  y = data.target
  return X, y

X, y = get_dataset(dataset_name)
st.write("Shape of dataset: ", X.shape)
st.write("Number of classes: ", len(np.unique(y)))

def add_parameter_ui(clf_name):
  params = dict()
  if clf_name == "K-Nearest Neighbors":
    K = st.sidebar.slider("K", 1, 15)
    params["K"] = K
  elif clf_name == "Support Vector Machine":
    C = st.sidebar.slider("C", 0.01, 10.0)
    kernel = st.sidebar.radio("Pick a kernel", ("linear", "poly", "rbf", "sigmoid", "precomputed"))
    params["C"] = C
    params["kernel"] = kernel
  elif clf_name == "Random Forest":
    max_depth = st.sidebar.slider("max_depth", 2, 15)
    n_estimators = st.sidebar.slider("n_estimators", 1, 100)
    params["max_depth"] = max_depth
    params["n_estimators"] = n_estimators
  elif clf_name == "Decision Tree":
    max_depth = st.sidebar.slider("max_depth", 2, 15)
    params["max_depth"] = max_depth
  elif clf_name == "Gradient Boosting":
    n_estimators = st.sidebar.slider("n_estimators", 1, 100)
    learning_rate = st.sidebar.slider("learning_rate", 0.01, 1.0, step=0.1)
    params["n_estimators"] = n_estimators
    params["learning_rate"] = learning_rate
  elif clf_name == "AdaBoost":
    n_estimators = st.sidebar.slider("n_estimators", 1, 100)
    learning_rate = st.sidebar.slider("learning_rate", 0.01, 1.0, step=0.1)
    params["n_estimators"] = n_estimators
    params["learning_rate"] = learning_rate
  elif clf_name == "Gaussian Naive Bayes":
    pass
  elif clf_name == "Neural Network (MLP)":
    activation_function = st.sidebar.radio("Pick an activation", ("identity", "logistic", "tanh", "relu"))
    solver = st.sidebar.radio("Pick a solver", ("lbfgs", "sgd", "adam"))
    learning_rate = st.sidebar.radio("Pick a learning rate method", ("constant", "invscaling", "adaptive"))
    params["activation"] = activation_function
    params["solver"] = solver
    params["learning_rate"] = learning_rate

  return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
  if clf_name == "K-Nearest Neighbors":
    clf = KNeighborsClassifier(n_neighbors=params["K"])
  elif clf_name == "Support Vector Machine":
    clf = SVC(C=params["C"], kernel=params["kernel"])
  elif clf_name == "Random Forest":
    clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                 max_depth=params["max_depth"], random_state=1234)
  elif clf_name == "Decision Tree":
    clf = DecisionTreeClassifier(max_depth=params["max_depth"], random_state=1234)
  elif clf_name == "Gradient Boosting":
    clf = GradientBoostingClassifier(n_estimators=params["n_estimators"], 
                                     learning_rate=params["learning_rate"], random_state=1234)
  elif clf_name == "AdaBoost":
    clf = GradientBoostingClassifier(n_estimators=params["n_estimators"], 
                                     learning_rate=params["learning_rate"], random_state=1234)
  elif clf_name == "Gaussian Naive Bayes":
    clf = GaussianNB()
  elif clf_name == "Neural Network (MLP)":
    clf = MLPClassifier(activation=params["activation"], solver=params["solver"], learning_rate=params["learning_rate"], random_state=1234)
  return clf

clf = get_classifier(classifier_name, params)

#Classification section
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.write(f"Classifier = {classifier_name}")
st.write(f"Accuracy = {acc}")

#Plotting section
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.7, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

st.pyplot(fig)

