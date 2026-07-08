# Canada Per Capita Income Prediction

## Overview
This project builds a simple **Linear Regression** model to predict the per capita income of Canadian citizens for the year 2020. The model is trained on historical income data from 1970 to 2016.

This was completed as part of a machine learning exercise to practice data handling with Pandas and model building with Scikit-Learn.

## Project Structure
* `canada_per_capita_income.csv`: The historical dataset containing two columns: `year` and `per capita income (US$)`.
* `predict.py`: The main Python script that loads the data, trains the linear regression model, and outputs the prediction.
* `requirements.txt`: The list of Python dependencies required to run the project.

## Requirements
This project uses the following libraries:
* `pandas` (for data manipulation)
* `scikit-learn` (for the machine learning model)

## Installation & Setup
This project is optimized to run with `uv`, a blazingly fast Python package manager.

1.  Ensure you have your files (`canada_per_capita_income.csv`, `predict.py`, and `requirements.txt`) in the same directory.
2.  Run the script instantly using `uv` without manually setting up a virtual environment:
    ```bash
    uv run --with-requirements requirements.txt python predict.py
    ```
    *Alternatively, you can install the dependencies into your environment using `uv pip install -r requirements.txt` and then run `python predict.py`.*

## Results
Based on the historical trend, the linear regression model predicts the following per capita income for Canadian citizens in **2020**:

**Predicted Income: $41,288.69**
