import re

filepath = 'D:/hacker/energivanu/Energivanu2/magazine/build_magazine.py'
with open(filepath, 'r') as f:
    content = f.read()

# Let's fix text contrast where WHITE text was written on light backgrounds:

# 1. section_title (Line 73 approx): change self.set_text_color(*WHITE) to self.set_text_color(*BLACK)
# Wait, let's look at the function:
# def section_title(self, title, subtitle=''):
#     self.set_font('helvetica', 'B', 26)
#     self.set_text_color(*WHITE)
content = re.sub(
    r"def section_title\(self, title, subtitle=''\):\n\s+self\.set_font\('helvetica', 'B', 26\)\n\s+self\.set_text_color\(\*WHITE\)",
    "def section_title(self, title, subtitle=''):\n        self.set_font('helvetica', 'B', 26)\n        self.set_text_color(*BLACK)",
    content
)

# 2. heading2 (Line 90 approx): change self.set_text_color(*WHITE) to self.set_text_color(*BLACK)
content = re.sub(
    r"def heading2\(self, text, y=None\):\n\s+if y: self\.set_y\(y\)\n\s+self\.set_font\('helvetica', 'B', 15\)\n\s+self\.set_text_color\(\*WHITE\)",
    "def heading2(self, text, y=None):\n        if y: self.set_y(y)\n        self.set_font('helvetica', 'B', 15)\n        self.set_text_color(*BLACK)",
    content
)

# 3. simple_table highlight row text color: change self.set_text_color(*WHITE) to self.set_text_color(*BLACK)
content = content.replace("self.set_text_color(*WHITE)\n                self.set_font('helvetica', 'B', 8)",
                          "self.set_text_color(*BLACK)\n                self.set_font('helvetica', 'B', 8)")

# 4. Cover page logo text color: change pdf.set_text_color(*WHITE) to pdf.set_text_color(*BLACK)
content = content.replace("pdf.set_text_color(*WHITE)\n    pdf.set_xy(10, 2)\n    pdf.cell(120, 8, 'ENERGIVANU INSIGHTS'",
                          "pdf.set_text_color(*BLACK)\n    pdf.set_xy(10, 2)\n    pdf.cell(120, 8, 'ENERGIVANU INSIGHTS'")

# 5. Cover page main headline: change pdf.set_text_color(*WHITE) to pdf.set_text_color(*BLACK)
content = content.replace("pdf.set_text_color(*WHITE)\n    pdf.set_xy(20, 95)\n    pdf.multi_cell(170, 14, 'The Open-Source Engine",
                          "pdf.set_text_color(*BLACK)\n    pdf.set_xy(20, 95)\n    pdf.multi_cell(170, 14, 'The Open-Source Engine")

# 6. TOC page titles: change pdf.set_text_color(*WHITE) to pdf.set_text_color(*BLACK)
content = content.replace("pdf.set_text_color(*WHITE)\n        pdf.set_xy(38, y)\n        pdf.cell(130, 8, title)",
                          "pdf.set_text_color(*BLACK)\n        pdf.set_xy(38, y)\n        pdf.cell(130, 8, title)")

# 7. Competitive matrix header text color in table is CYAN (fine) but let's check highlight row:
# Highlight row in simple_table has already been handled in step 3.

# 8. Roadmap titles: change pdf.set_text_color(*WHITE) to pdf.set_text_color(*BLACK)
content = content.replace("pdf.set_text_color(*WHITE)\n        pdf.set_xy(48, y)\n        pdf.cell(120, 6, title)",
                          "pdf.set_text_color(*BLACK)\n        pdf.set_xy(48, y)\n        pdf.cell(120, 6, title)")

# 9. Founder name in Founder box: change pdf.set_text_color(*WHITE) to pdf.set_text_color(*BLACK)
content = content.replace("pdf.set_text_color(*WHITE)\n    pdf.set_xy(52, y + 5)\n    pdf.cell(100, 7, 'Ved Kumar')",
                          "pdf.set_text_color(*BLACK)\n    pdf.set_xy(52, y + 5)\n    pdf.cell(100, 7, 'Ved Kumar')")

with open(filepath, 'w') as f:
    f.write(content)

print("Contrast issues resolved.")
