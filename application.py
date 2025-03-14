from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)
model = pickle.load(open("LinearRegressionModel.pkl", "rb"))

# Load the cleaned CSV file
df = pd.read_csv("Cleaned_Car_data.csv")

# Get unique companies, years, and fuel types
companies = df["company"].unique().tolist()
years = sorted(df["year"].unique(), reverse=True)
fuel_types = df["fuel_type"].unique().tolist()

@app.route("/")
def index():
    return render_template("index.html", companies=companies, years=years, fuel_type=fuel_types)

@app.route("/get_models", methods=["POST"])
def get_models():
    try:
        data = request.get_json()
        selected_company = data.get("company")

        if not selected_company:
            return jsonify({"error": "No company selected"}), 400

        models = df[df["company"] == selected_company]["name"].unique().tolist()

        if not models:
            return jsonify({"models": []})  # No models found for the company

        return jsonify({"models": models})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Internal Server Error
    
@app.route('/predict', methods=['POST'])
def predict():
    company= request.form.get('company')

    car_model = request.form.get('car_model')
    year=int(request.form.get('year'))
    fuel_type=request.form.get('fuel_type')
    kms_driven=int(request.form.get('kilo_driven'))
    print(company, car_model, year, fuel_type, kms_driven)

    prediction = model.predict(pd.DataFrame([[car_model, company,year, kms_driven, fuel_type]], columns=['name','company','year','kms_driven', 'fuel_type']))
    print(prediction)
    return ""

if __name__ == "__main__":
    app.run(debug=True)