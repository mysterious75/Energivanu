import re

with open('D:/hacker/energivanu/Energivanu2/magazine/build_magazine.py', 'r') as f:
    content = f.read()

# Fix numbering
content = content.replace("pdf.top_bar('Energivanu', 2)", "pdf.top_bar('Energivanu', 1)")
content = content.replace("pdf.top_bar('The Problem', 3)", "pdf.top_bar('The Problem', 2)")
content = content.replace("pdf.top_bar('The Solution', 4)", "pdf.top_bar('The Solution', 3)")
content = content.replace("pdf.top_bar('Technology', 5)", "pdf.top_bar('Technology', 4)")
content = content.replace("pdf.top_bar('Training', 6)", "pdf.top_bar('Training', 5)")
content = content.replace("pdf.top_bar('Validation', 7)", "pdf.top_bar('Validation', 6)")
content = content.replace("pdf.top_bar('Competition', 8)", "pdf.top_bar('Competition', 7)")
content = content.replace("pdf.top_bar('Market', 9)", "pdf.top_bar('Market', 8)")
content = content.replace("pdf.top_bar('Future', 10)", "pdf.top_bar('Future', 9)")

# Fix page 4 overflow
p4_split = """    y = pdf.heading2('Three Engines, One Pipeline')"""
p4_fix = """    pdf.footer_bar()
    pdf.dark_page()
    pdf.top_bar('The Solution - Cont.', 4)
    y = 20
    y = pdf.heading2('Three Engines, One Pipeline', y)"""
content = content.replace(p4_split, p4_fix)

# Fix page 5 overflow
p5_split = """    y = pdf.heading2('BESS Physics', y)"""
p5_fix = """    pdf.footer_bar()
    pdf.dark_page()
    pdf.top_bar('Technology - Cont.', 5)
    y = 20
    y = pdf.heading2('BESS Physics', y)"""
content = content.replace(p5_split, p5_fix)

# Fix page 7 overflow
p7_split = """    y = pdf.heading2('Real Hardware Validation', y)"""
p7_fix = """    pdf.footer_bar()
    pdf.dark_page()
    pdf.top_bar('Validation - Cont.', 7)
    y = 20
    y = pdf.heading2('Real Hardware Validation', y)"""
content = content.replace(p7_split, p7_fix)

# Fix page 8 overflow
p8_split = """    y = pdf.heading2('The Open-Source Advantage', y)"""
p8_fix = """    pdf.footer_bar()
    pdf.dark_page()
    pdf.top_bar('Competition - Cont.', 8)
    y = 20
    y = pdf.heading2('The Open-Source Advantage', y)"""
content = content.replace(p8_split, p8_fix)

# Fix page 9 overflow
p9_split = """    y = pdf.heading2('Revenue Model', pdf.get_y() + 2)"""
p9_fix = """    pdf.footer_bar()
    pdf.dark_page()
    pdf.top_bar('Market - Cont.', 9)
    y = 20
    y = pdf.heading2('Revenue Model', y)"""
content = content.replace(p9_split, p9_fix)

# Fix page 10 overflow
p10_split = """    y = pdf.heading2('The Vision', y + 2)"""
p10_fix = """    pdf.footer_bar()
    pdf.dark_page()
    pdf.top_bar('Future - Cont.', 10)
    y = 20
    y = pdf.heading2('The Vision', y)"""
content = content.replace(p10_split, p10_fix)

# Since we added pages, the total page count shifts. But FPDF doesn't mind.
with open('D:/hacker/energivanu/Energivanu2/magazine/build_magazine.py', 'w') as f:
    f.write(content)

print("Magazine builder updated successfully.")
