import gradio as gr
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Load trained models
model_credit = joblib.load('credit_model.pkl')
model_beauty = joblib.load('beauty_model.pkl')

def predict_income(age, debt_ratio, open_lines):
    pred = model_credit.predict([[age, debt_ratio, open_lines]])
    return f"${pred[0]:,.2f} / month"

def predict_wage(exper, educ, looks):
    pred = model_beauty.predict([[exper, educ, looks]])
    return f"${pred[0]:.2f} / hour"

# Modern custom CSS for rounded glassmorphism-inspired cards
custom_css = """
.container { max-width: 900px; margin: auto; }
.card-header { font-weight: 700; font-size: 1.15rem; color: #4f46e5; margin-bottom: 0.5rem; }
.submit-btn { background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%) !important; color: white !important; font-weight: 600; border-radius: 8px !important; }
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
)

with gr.Blocks(theme=theme, css=custom_css, title="AI Analytics Hub") as demo:
    with gr.Column(elem_classes=["container"]):
        gr.Markdown(
            """
            # 📊 Regression Analytics Dashboard
            Predict metrics using real-time machine learning models.
            """
        )
        
        with gr.Tabs():
            # Tab 1: Credit Scoring
            with gr.TabItem("💳 Income Estimator (Credit Data)"):
                gr.Markdown("Estimate borrower **Monthly Income** using credit profile indicators.")
                with gr.Group():
                    with gr.Row():
                        age = gr.Slider(minimum=18, maximum=100, value=35, step=1, label="Applicant Age")
                        debt_ratio = gr.Number(value=0.35, label="Debt-to-Income Ratio (e.g., 0.35)", precision=2)
                    open_lines = gr.Slider(minimum=0, maximum=50, value=6, step=1, label="Active Credit Lines & Loans")
                    
                    btn_credit = gr.Button("Calculate Expected Income", elem_classes=["submit-btn"])
                    out_credit = gr.Textbox(label="Predicted Monthly Income", interactive=False)
                    btn_credit.click(predict_income, inputs=[age, debt_ratio, open_lines], outputs=out_credit)

            # Tab 2: Beauty / Wage
            with gr.TabItem("💼 Hourly Wage Predictor (Beauty Data)"):
                gr.Markdown("Estimate expected **Hourly Wage** based on education, experience, and ratings.")
                with gr.Group():
                    with gr.Row():
                        exper = gr.Slider(minimum=0, maximum=50, value=10, step=1, label="Years of Work Experience")
                        educ = gr.Slider(minimum=0, maximum=24, value=12, step=1, label="Years of Formal Education")
                    looks = gr.Radio(choices=[1, 2, 3, 4, 5], value=3, label="Rating Assessment (Scale 1-5)")
                    
                    btn_wage = gr.Button("Calculate Estimated Wage", elem_classes=["submit-btn"])
                    out_wage = gr.Textbox(label="Predicted Hourly Rate", interactive=False)
                    btn_wage.click(predict_wage, inputs=[exper, educ, looks], outputs=out_wage)

if __name__ == "__main__":
    demo.launch()