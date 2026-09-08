from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# --------------------------------------------------
# Load trained ML files
# --------------------------------------------------

model = joblib.load("heart_disease_knn.pkl")
scaler = joblib.load("heart_disease_scaler.pkl")
expected_columns = joblib.load("heart_disease_features.pkl")


# --------------------------------------------------
# Home Page
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# Prediction
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get values from HTML form

        age = int(request.form["age"])
        sex = request.form["sex"]
        chest_pain = request.form["chest_pain"]

        resting_bp = float(
            request.form["resting_bp"]
        )

        cholesterol = float(
            request.form["cholesterol"]
        )

        fasting_bs = int(
            request.form["fasting_bs"]
        )

        resting_ecg = request.form["resting_ecg"]

        max_hr = int(
            request.form["max_hr"]
        )

        exercise_angina = request.form["exercise_angina"]

        oldpeak = float(
            request.form["oldpeak"]
        )

        st_slope = request.form["st_slope"]


        # --------------------------------------------------
        # Create input dictionary
        # --------------------------------------------------

        raw_input = {
            "Age": age,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "MaxHR": max_hr,
            "Oldpeak": oldpeak
        }


        # --------------------------------------------------
        # Add encoded categorical variables
        # --------------------------------------------------

        categorical_data = {
            "Sex": sex,
            "ChestPainType": chest_pain,
            "RestingECG": resting_ecg,
            "ExerciseAngina": exercise_angina,
            "ST_Slope": st_slope
        }


        for feature, value in categorical_data.items():

            column_name = feature + "_" + value

            if column_name in expected_columns:

                raw_input[column_name] = 1


        # --------------------------------------------------
        # Convert to DataFrame
        # --------------------------------------------------

        input_df = pd.DataFrame([raw_input])


        # --------------------------------------------------
        # Add missing columns
        # --------------------------------------------------

        for column in expected_columns:

            if column not in input_df.columns:

                input_df[column] = 0


        # --------------------------------------------------
        # Arrange columns exactly like training data
        # --------------------------------------------------

        input_df = input_df[expected_columns]


        # --------------------------------------------------
        # Scale input
        # --------------------------------------------------

        scaled_input = scaler.transform(input_df)


        # --------------------------------------------------
        # Prediction
        # --------------------------------------------------

        prediction = model.predict(scaled_input)[0]


        # --------------------------------------------------
        # Probability
        # --------------------------------------------------

        # KNN confidence: percentage of the 5 nearest neighbors
        # belonging to the predicted class.
        confidence = None
        neighbor_count = getattr(model, "n_neighbors", None)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled_input)[0]
            confidence = float(max(probabilities))


        # --------------------------------------------------
        # Result
        # --------------------------------------------------

        if prediction == 1:

            result = "⚠️ High Risk of Heart Disease"

            result_class = "high-risk"

        else:

            result = "✅ Low Risk of Heart Disease"

            result_class = "low-risk"


        return render_template(
            "index.html",
            result=result,
            result_class=result_class,
            confidence=confidence,
            neighbor_count=neighbor_count
        )


    except Exception as error:

        return render_template(
            "index.html",
            result=f"Error: {error}",
            result_class="error"
        )


# --------------------------------------------------
# Run Application
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )