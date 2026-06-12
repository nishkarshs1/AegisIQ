import io
from fpdf import FPDF
from datetime import datetime

class PDF(FPDF):
    def header(self):
        # Header text
        self.set_font("helvetica", "B", 24)
        self.set_text_color(108, 174, 249)  # #6CAEF9 Sky Blue
        self.cell(0, 10, "AegisIQ", border=0, new_x="LMARGIN", new_y="NEXT", align="L")
        self.set_font("helvetica", "I", 12)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, "Intelligent Risk Assessment Report", border=0, new_x="LMARGIN", new_y="NEXT", align="L")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def generate_risk_report(user_details: dict, prediction: str, prob: float, theme: dict) -> bytes:
    pdf = PDF()
    pdf.add_page()
    
    # Date
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(150, 150, 150)
    pdf.set_xy(10, 20)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="R")
    pdf.set_xy(10, 50)
    
    # Title
    pdf.set_font("helvetica", "B", 18)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 10, "Assessment Results", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)
    
    # Prediction Block
    pdf.set_font("helvetica", "B", 16)
    # Convert hex to RGB for the theme color
    hex_color = theme[prediction]["color"].lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    pdf.set_text_color(r, g, b)
    pdf.cell(0, 10, f"Risk Category: {prediction.upper()} RISK", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("helvetica", "", 14)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, f"Confidence Score: {prob:.1f}%", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(15)
    
    # Details Table
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 10, "Provided Details", new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.ln(2)
    
    pdf.set_font("helvetica", "", 12)
    # Background for alternating rows
    fill = False
    for key, value in user_details.items():
        pdf.set_text_color(100, 100, 100)
        pdf.set_fill_color(245, 245, 245)
        
        # Clean up keys for display
        display_key = key.replace('_', ' ').title()
        if display_key == "Income Lpa": display_key = "Annual Income (LPA)"
        
        pdf.cell(60, 12, f"  {display_key}", border=1, fill=fill)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 12, f"  {value}", border=1, new_x="LMARGIN", new_y="NEXT", fill=fill)
        fill = not fill
        
    pdf.ln(10)
    
    # Derived Metrics & Analysis
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 10, "Key Health Indicators & Analysis", new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.ln(2)
    
    try:
        bmi = float(user_details.get("weight", 0)) / (float(user_details.get("height", 1)) ** 2)
        bmi_text = f"Calculated BMI: {bmi:.1f} (Body Mass Index is a core indicator of baseline health risk.)"
    except:
        bmi_text = "Calculated BMI: N/A"
        
    pdf.set_font("helvetica", "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 6, bmi_text)
    
    if prediction == "Low":
        explanation = "A 'Low' risk category indicates that your demographic and lifestyle factors fall into the safest percentiles. Expected premiums are at standard or preferred rates."
    elif prediction == "Medium":
        explanation = "A 'Medium' risk category indicates moderate risk factors (such as higher age, or slightly elevated BMI). Your premiums may be slightly adjusted."
    else:
        explanation = "A 'High' risk category strongly indicates compounded risk factors, such as combining smoking with an elevated BMI, or advanced age brackets. This significantly increases premium costs."

    pdf.ln(4)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 6, explanation)
    
    pdf.ln(15)
    
    pdf.set_font("helvetica", "I", 10)
    pdf.set_text_color(150, 150, 150)
    disclaimer = "Disclaimer: This assessment is generated using a Machine Learning model (AegisIQ) based on the provided inputs. It is intended for informational purposes and should not be considered final medical or financial underwriting advice."
    pdf.multi_cell(0, 6, disclaimer)
    
    return bytes(pdf.output())
