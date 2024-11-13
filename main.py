import streamlit as st  # Importing the streamlit library for creating the web app
from sklearn import datasets  # Importing datasets from sklearn
import numpy as np  # Importing numpy for numerical operations

from sklearn.neighbors import KNeighborsClassifier  # Importing KNN classifier
from sklearn.svm import SVC  # Importing SVM classifier
from sklearn.ensemble import RandomForestClassifier  # Importing Random Forest classifier
from sklearn.model_selection import train_test_split  # Importing train_test_split for splitting the dataset
from sklearn.metrics import accuracy_score  # Importing accuracy_score for evaluating the model
from sklearn.decomposition import PCA  # Importing PCA for dimensionality reduction
import matplotlib.pyplot as plt  # Importing matplotlib for plotting

st.title('Streamlit App Demo')  # Setting the title of the Streamlit app
st.write("""Explore different features of Streamlit""")  # Writing a description in the Streamlit app

# Creating a sidebar dropdown for selecting the dataset
dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine"))
# Creating a sidebar dropdown for selecting the classifier
classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))

# Function to load the selected dataset
def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data  # Features
    y = data.target  # Target variable
    return X, y

X, y = get_dataset(dataset_name)  # Loading the selected dataset
st.write("Shape of dataset:", X.shape)  # Displaying the shape of the dataset
st.write("Number of classes:", len(np.unique(y)))  # Displaying the number of classes

# Function to add classifier-specific parameters to the sidebar
def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)  # Slider for selecting K in KNN
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)  # Slider for selecting C in SVM
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("Max Depth", 2, 15)  # Slider for selecting max depth in Random Forest
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)  # Slider for selecting number of estimators in Random Forest
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

params = add_parameter_ui(classifier_name)  # Getting the parameters for the selected classifier

# Function to get the selected classifier with the specified parameters
def get_classfier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:        
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], 
                                     max_depth=params["max_depth"], 
                                     random_state=1234)
    return clf

clf = get_classfier(classifier_name, params)

#Classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.write(f"Classifier: {classifier_name}")
st.write(f"Accuracy: {acc}")

#Plot
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

st.pyplot(fig)