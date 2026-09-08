HEART DISEASE PREDICTION - FLASK PROJECT

1. Open this folder in VS Code.
2. Open the VS Code terminal in this folder.
3. Install dependencies:
   pip install -r requirements.txt

4. Start the website:
   python app.py

5. Open in Chrome:
   http://127.0.0.1:5000

PROJECT STRUCTURE
-----------------
app.py
heart.csv
heart_disease_knn.pkl
heart_disease_scaler.pkl
heart_disease_features.pkl
requirements.txt
README.txt
templates/
    index.html

IMPORTANT
---------
The .pkl files are the trained model artifacts. Keep their filenames unchanged.
The website is an ML demonstration and is not medical advice.


MODEL OUTPUT
------------
The website uses the saved KNN model directly for prediction.
The displayed "KNN Model Confidence" is the highest class probability
returned by KNeighborsClassifier, i.e. the proportion of nearest
neighbors supporting the predicted class. It is not a clinical probability
or medical diagnosis.
