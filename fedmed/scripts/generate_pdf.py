from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import textwrap
from pathlib import Path
import re

html_path = Path(r"C:\Users\user\OneDrive\Desktop\Axlero Internship\FedMed\fedmed-federated-learning.worktrees\fedmed-federated-learning-privacy-ml\fedmed\docs\one_pager.html")
pdf_path = html_path.with_suffix('.pdf')

if not html_path.exists():
    raise SystemExit('HTML one_pager not found')

html = html_path.read_text(encoding='utf-8')
# Remove scripts/styles and tags
text = re.sub(r'<script[\s\S]*?</script>', '', html, flags=re.I)
text = re.sub(r'<style[\s\S]*?</style>', '', text, flags=re.I)
text = re.sub(r'<[^>]+>', '', text)
text = re.sub(r'\s+',' ', text).strip()

# Wrap and limit lines to approximate one page
lines = textwrap.wrap(text, width=100)
max_lines = 60
lines = lines[:max_lines]

c = canvas.Canvas(str(pdf_path), pagesize=letter)
width, height = letter
c.setFont('Helvetica', 10)
margin = 72
y = height - margin
for line in lines:
    c.drawString(margin, y, line)
    y -= 12
    if y < margin:
        c.showPage()
        c.setFont('Helvetica', 10)
        y = height - margin

c.save()
print('Wrote', pdf_path)
