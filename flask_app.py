from flask import Flask, request, render_template
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# Load the trained models
model_credit = joblib.load('credit_model.pkl')
model_beauty = joblib.load('beauty_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_income', methods=['POST'])
def predict_income():
    age = float(request.form['age'])
    debt_ratio = float(request.form['debt_ratio'])
    open_lines = float(request.form['open_lines'])
    
    pred = model_credit.predict([[age, debt_ratio, open_lines]])
    result = f"${pred[0]:.2f} / month"
    
    return render_template('index.html', result_credit=result)

@app.route('/predict_wage', methods=['POST'])
def predict_wage():
    exper = float(request.form['exper'])
    educ = float(request.form['educ'])
    looks = float(request.form['looks'])
    
    pred = model_beauty.predict([[exper, educ, looks]])
    result = f"${pred[0]:.2f} / hour"
    
    return render_template('index.html', result_wage=result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)