"""Fill the two student's declarations using the exact details supplied by the student.
Generate Times New Roman 12pt PDFs on Windows. Typed signature, NOT handwritten.
"""
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/"Declarations"
NAME="Sabareesh S"
REGISTER="127003218"
UNIVERSITY="SASTRA Deemed to be University"
DEPARTMENT="Computer Science and Engineering"
PLACE="Thanjavur"
DATE="20/10/2026"
TITLE="AUTOSAR HLD AI Assistant: Grounded AI for High-Level Design Analysis, Engineering Q&A and Architecture Exploration"

FONT_DIR=Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("TimesNewRoman",str(FONT_DIR/"times.ttf")))
pdfmetrics.registerFont(TTFont("TimesNewRomanBold",str(FONT_DIR/"timesbd.ttf")))
body=ParagraphStyle("Body",fontName="TimesNewRoman",fontSize=12,leading=18,spaceAfter=9)
bold=ParagraphStyle("Bold",parent=body,fontName="TimesNewRomanBold",spaceBefore=6)
def para(s,em=False):
    return Paragraph(escape(s),bold if em else body)
def common(title):
    elements=[para(title,True),Spacer(1,6),
        para("Project: "+TITLE),para("Name: "+NAME),
        para("Register No.: "+REGISTER),
        para("University: "+UNIVERSITY),
        para("Department: "+DEPARTMENT),Spacer(1,10)]
    return elements
def save(filename,story):
    output=DEST/filename
    doc=SimpleDocTemplate(str(output),pagesize=A4,topMargin=21*mm,bottomMargin=19*mm,leftMargin=21*mm,rightMargin=21*mm,title=filename.replace("_"," ").replace(".pdf",""),author=NAME)
    doc.build(story)
    print("Created:",output.relative_to(ROOT))
def footer():
    return [Spacer(1,14),para("Place: "+PLACE),para("Date: "+DATE),
        para("Signature (typed, as supplied by student): "+NAME)]
academic=common("Academic Integrity Declaration")
academic += [
    para("I declare that this submission accurately describes my own work and contributions. External source code, datasets, pretrained models, and supporting references have been credited. Assistance from generative AI and coding agents is disclosed in the accompanying AI Tool Usage Declaration."),
    para("I have reviewed the project's behavior and test evidence to the extent stated in the report. I have not knowingly fabricated performance measurements or falsely represented model training."),
    para("This declaration is based on the student's supplied particulars. The typed signature below is not a handwritten or independently authenticated signature."),
]+footer()
save("Academic_Integrity_Declaration.pdf",academic)
ai=common("AI Tool Usage Declaration")
ai += [
    para("I disclose the following tools and models used or discussed during the development of this capstone. Their outputs must be verified against the actual submitted code, documentation, and evaluation results before final academic acceptance."),
]
rows=[
    ["Tool / Model","Purpose and affected artifact"],
    ["Google Antigravity / coding agents","Planning and assisted implementation of Python code, tests, and frontend"],
    ["ChatGPT","Concept explanations, project prompts, synthetic samples, and academic documentation drafting"],
    ["all-MiniLM-L6-v2","Pretrained sentence embeddings used for document and query vectors"],
    ["Google Gemini API","LLM-based engineering question-answer generation from retrieved context"],
    ["MockLLMProvider","Deterministic offline test double for integration and tests"],
]
cells=[[Paragraph(escape(cell),body) for cell in row] for row in rows]
t=Table(cells,colWidths=[55*mm,110*mm],repeatRows=1,hAlign="LEFT")
t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),.5,colors.grey),
    ("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
ai += [t,Spacer(1,12),
    para("Student verification: The implementation is subject to source-code inspection, unit/integration tests, and manual evaluation of generated answers and source citations. A valid citation ID alone is not proof of factual correctness."),
    para("I take responsibility for reviewing the AI-assisted work and complying with the institution's academic integrity and AI-use regulations."),
    para("The typed signature below is provided as text requested by the student, not a simulated handwritten signature."),
] + footer()
save("AI_Tool_Usage_Declaration.pdf",ai)
