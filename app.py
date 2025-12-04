from flask import Flask, request, render_template
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("house_price_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect input values from form
        input_data = {
            "OverallQual": [float(request.form["OverallQual"])],
            "GrLivArea": [float(request.form["GrLivArea"])],
            "GarageCars": [float(request.form["GarageCars"])],
            "TotalBsmtSF": [float(request.form["TotalBsmtSF"])],
            "Neighborhood": [request.form["Neighborhood"]]
        }

        # Convert to DataFrame
        df = pd.DataFrame(input_data)

        # Predict price
        prediction = model.predict(df)[0]

        return render_template("index.html",
                               prediction_text=f"🏡 Predicted House Price: ${prediction:,.2f}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"⚠️ Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
 