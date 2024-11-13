# StreamlitDataApp

![StreamlitDataGIF](https://github.com/AndrewLamCS/StreamlitDataApp/blob/main/Kapture%202024-11-13%20at%2016.33.07.gif)

This is a simple **Streamlit** app that demonstrates the use of different machine learning classifiers (KNN, SVM, and Random Forest) on various datasets, including Iris, Breast Cancer, and Wine datasets from `sklearn`. The app also includes a **PCA (Principal Component Analysis)** visualization of the dataset to show the reduction of dimensions for the chosen dataset.

To run this app, you'll need the following Python packages installed: `streamlit`, `scikit-learn`, `numpy`, and `matplotlib`. You can install these dependencies using `pip`:

```bash
pip install streamlit scikit-learn numpy matplotlib
```
## How to Run the App
Clone the repository to your local machine:

```bash

git clone https://github.com/AndrewLamCS/StreamlitDataApp.git
cd StreamlitDataApp
```
## Create and activate a virtual environment (optional but recommended):

```bash

python3 -m venv .venv
source .venv/bin/activate  # On Windows, use .venv\Scripts\activate

```
## Install the required dependencies:
```
bash
pip install -r requirements.txt

```
## Run the Streamlit app:
```
bash
streamlit run main.py
This will open the app in your default web browser. You can interact with the app by selecting different datasets, classifiers, and adjusting the classifier parameters through the sidebar.
```
## Project Structure
```
bash
StreamlitDataApp/
│
├── main.py               # The main Streamlit app file
├── requirements.txt      # List of dependencies
├── README.md             # Project documentation
└── .venv/                # Virtual environment (optional)
This project is open-source and available under the MIT License.
```
