from flask import Flask, render_template, request
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for the index (landing) page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the prediction page
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score'))
            )

            pred_df = data.get_data_as_dataframe()
            print("Input DataFrame:\n", pred_df)

            predict_pipeline = PredictPipeline()
            result = predict_pipeline.predict(pred_df)

            return render_template('home.html', results=result[0])

        except Exception as e:
            return f"Something went wrong: {e}"

if __name__ == "__main__":
    app.run(host="0.0.0.0")
