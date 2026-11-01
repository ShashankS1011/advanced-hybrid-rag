import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def generate_sample_financial_report():
    os.makedirs("data", exist_ok=True)
    pdf_path = "data/financial_report.pdf"

    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1E3A8A"),
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1E40AF"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = styles["Normal"]
    body_style.spaceAfter = 8

    # Title
    story.append(Paragraph("Acme Corporation - Quarterly Financial Report (Q4 2025)", title_style))
    story.append(Paragraph("<b>Author:</b> Financial Intelligence Unit | <b>Date:</b> January 2026", body_style))
    story.append(Spacer(1, 12))

    # Executive Summary
    story.append(Paragraph("1. Executive Summary", heading_style))
    story.append(
        Paragraph(
            "Acme Corporation recorded a total net revenue of $12.5 Million in Q4 2025, representing a 14% year-over-year increase. "
            "The primary drivers of growth were Cloud Services adoption and AI enterprise contract renewals. Operating expenses "
            "stayed controlled at $7.2 Million, yielding an EBITDA margin of 28.5%.",
            body_style,
        )
    )

    # Financial Highlights Table
    story.append(Paragraph("2. Financial Metrics Summary", heading_style))
    data = [
        ["Metric", "Q4 2024", "Q4 2025", "YoY Change"],
        ["Total Revenue", "$10.9M", "$12.5M", "+14.6%"],
        ["Gross Margin", "62.0%", "65.4%", "+3.4%"],
        ["Operating Expenses", "$6.8M", "$7.2M", "+5.8%"],
        ["Net Income", "$2.1M", "$3.1M", "+47.6%"],
        ["Cash Reserves", "$18.4M", "$22.0M", "+19.5%"],
    ]
    t = Table(data, colWidths=[150, 100, 100, 100])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 12))

    # Risk Assessment
    story.append(Paragraph("3. Operational Risks & Future Outlook", heading_style))
    story.append(
        Paragraph(
            "Key risks identified for FY2026 include cloud vendor supply chain constraints and increased infrastructure electricity costs. "
            "Management plans to hedge electricity costs and diversify server hardware deployment across multiple regional cloud providers.",
            body_style,
        )
    )

    doc.build(story)
    print(f"Sample PDF successfully created at: {pdf_path}")


if __name__ == "__main__":
    generate_sample_financial_report()
