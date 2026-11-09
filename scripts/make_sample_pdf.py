"""Generate the sample PDF used for the demo and the evaluation.

The handbook describes a FICTIONAL company. It exists so the project has a
realistic, multi-page document with known facts on known pages - which is
exactly what you need to measure whether retrieval finds the right page.

Run:  python scripts/make_sample_pdf.py
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parent.parent / "sample_docs" / "brightwave_handbook.pdf"

# One entry per page: (heading, [paragraphs])
PAGES = [
    ("Brightwave Logistics - Employee Handbook (fictional sample)", [
        "Welcome to Brightwave Logistics. This handbook explains how we work, what you can expect "
        "from us and what we expect from you. It applies to all employees in Germany and Austria.",
        "Brightwave was founded in 2014 in Leipzig and moves temperature-controlled freight for "
        "pharmacies and food producers. Today we employ 340 people across four warehouses.",
        "If anything in this handbook is unclear, contact People Operations at people@brightwave.example. "
        "The handbook is reviewed every January; this is version 7.",
    ]),
    ("1. Working hours and remote work", [
        "Core hours are 10:00 to 15:00, Monday to Friday. Outside core hours you can plan your day freely, "
        "as long as your team knows when to reach you.",
        "Office staff may work remotely up to three days per week. Warehouse and driving roles are on-site "
        "by nature, but planners in those teams can work from home one day per week.",
        "Working from another country is allowed for up to 20 working days per calendar year and must be "
        "approved by your manager and People Operations at least two weeks in advance.",
        "Overtime is recorded in the time-tracking tool and is compensated with time off. Overtime above "
        "40 hours per quarter must be approved by your department lead.",
    ]),
    ("2. Vacation, sick leave and parental leave", [
        "Full-time employees receive 30 vacation days per year. In your first year the days are prorated "
        "from your start date. Up to five unused days can be carried over until 31 March of the next year.",
        "If you are sick, tell your manager before 9:00 on the first day. A doctor's certificate is required "
        "from the third consecutive sick day onward.",
        "Parents receive their statutory parental leave. In addition, Brightwave grants ten extra paid days "
        "off around the birth or adoption of a child, to be taken within the first six months.",
        "For your wedding or registered partnership you receive two additional paid days off.",
    ]),
    ("3. Pay, benefits and equipment", [
        "Salaries are paid on the last working day of each month. Salary bands are published internally and "
        "reviewed every April based on performance conversations held in March.",
        "Every employee receives a public transport ticket (Deutschlandticket) paid in full by the company, "
        "or alternatively a bike leasing subsidy of up to 60 euros per month.",
        "Each person has a yearly learning budget of 1,200 euros for courses, books and conferences, plus "
        "three paid learning days. Unused learning budget does not carry over.",
        "New office employees choose between a 14-inch laptop and a 16-inch laptop; laptops are replaced "
        "every four years. A home-office allowance of 300 euros is paid once in your first month.",
    ]),
    ("4. Travel and expenses", [
        "Business trips must be booked through the travel portal. Train travel is preferred for any trip "
        "under 600 kilometres; flights for shorter distances need approval from your department lead.",
        "Hotel costs are reimbursed up to 140 euros per night in most cities and up to 190 euros per night "
        "in Munich, Hamburg, Frankfurt, Vienna and Zurich.",
        "Meals during travel are covered by the statutory daily allowance. Alcohol is never reimbursed.",
        "Expense reports must be submitted within 30 days with digital receipts. Reports submitted later "
        "than 90 days after the expense cannot be reimbursed.",
    ]),
    ("5. IT security and data protection", [
        "Your password must have at least 14 characters and you must use the company password manager. "
        "Multi-factor authentication is mandatory for email, the VPN and all customer systems.",
        "Never copy customer data to private devices or private cloud storage. Customer temperature logs "
        "are retained for six years and then deleted automatically.",
        "Report a lost or stolen laptop or phone to the IT helpdesk within one hour, even outside working "
        "hours, by calling the 24/7 security line at extension 4400.",
        "Suspicious emails are forwarded to phishing@brightwave.example. Every employee completes a short "
        "security training each quarter.",
    ]),
    ("6. Health and safety in the warehouse", [
        "Safety shoes and high-visibility vests are mandatory in all warehouse areas, including for visitors "
        "and office staff passing through.",
        "Forklifts may only be driven by employees with a valid internal forklift licence, which is renewed "
        "every two years after a refresher course.",
        "Cold-storage rooms run at 2 to 8 degrees Celsius. Shifts inside a cold room are limited to 45 "
        "minutes, followed by a warm-up break of at least 15 minutes.",
        "Accidents and near misses are reported in the safety app on the same day. Each warehouse has two "
        "trained first aiders per shift.",
    ]),
    ("7. Onboarding, feedback and leaving", [
        "Every new hire gets an onboarding buddy for the first three months. The probation period is six "
        "months, with a review conversation after month three.",
        "Performance conversations happen twice a year, in March and September. Peer feedback is collected "
        "anonymously two weeks before each conversation.",
        "The notice period during probation is two weeks. After probation it follows your contract and the "
        "statutory rules, with a minimum of one month to the end of a calendar month.",
        "When you leave, return your laptop, badge and keys on your last working day. You receive a "
        "written reference letter (Arbeitszeugnis) within two weeks.",
    ]),
]


def build(path: Path = OUT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    story = []
    for i, (heading, paragraphs) in enumerate(PAGES):
        story.append(Paragraph(heading, styles["Heading1"]))
        story.append(Spacer(1, 8))
        for text in paragraphs:
            story.append(Paragraph(text, styles["BodyText"]))
            story.append(Spacer(1, 6))
        if i < len(PAGES) - 1:
            story.append(PageBreak())
    SimpleDocTemplate(str(path), pagesize=A4, title="Brightwave Handbook (fictional)").build(story)
    return path


if __name__ == "__main__":
    print(f"Wrote {build()}")
