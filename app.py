from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("loan_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Convert text inputs to numbers
    gender = 1 if request.form["gender"].strip().lower() == "male" else 0

    married = 1 if request.form["married"].strip().lower() == "yes" else 0

    dep = request.form["dependents"].strip()
    dependents = 3 if dep == "3+" else int(dep)

    education = 1 if request.form["education"].strip().lower() == "graduate" else 0

    self_employed = 1 if request.form["self_employed"].strip().lower() == "yes" else 0

    applicant_income = float(request.form["applicant_income"])

    coapplicant_income = float(request.form["coapplicant_income"])

    loan_amount = float(request.form["loan_amount"])

    loan_term = float(request.form["loan_term"])

    credit_history = 1 if request.form["credit_history"].strip().lower() == "good" else 0

    area = request.form["property_area"].strip().lower()

    if area == "rural":
        property_area = 0
    elif area == "semiurban":
        property_area = 1
    elif area == "urban":
        property_area = 2
    else:
        property_area = 0

    # Create feature array
    features = np.array([[gender, married, dependents, education,
                          self_employed, applicant_income,
                          coapplicant_income, loan_amount,
                          loan_term, credit_history,
                          property_area]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "✅Loan Approved"
    else:
        result = "❎Loan Rejected"

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)