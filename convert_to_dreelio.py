import re

filepath = 'D:/hacker/energivanu/Energivanu2/magazine/build_magazine.py'
with open(filepath, 'r') as f:
    content = f.read()

# Define Dreelio colors
dreelio_colors = """# Colors
BLACK = (26, 22, 21)        # Deep dark warm brown
DARK_BG = (244, 241, 238)   # Light gray-cream background
RED = (201, 80, 46)         # Burnt orange accent
CYAN = (21, 108, 194)       # Primary blue
GREEN = (14, 161, 88)       # Soft green
ORANGE = (201, 80, 46)      # Burnt orange accent
WHITE = (255, 255, 255)     # Clean white
GRAY = (117, 113, 112)      # Neutral gray
DARK_GRAY = (69, 63, 61)    # Warm dark brown
LIGHT_GRAY = (228, 226, 226)# Light border gray
MID_GRAY = (100, 100, 100)
"""

# Replace the color definitions
content = re.sub(r"# Colors.*?MID_GRAY = \(100, 100, 100\)", dreelio_colors, content, flags=re.DOTALL)

# Update helper functions for Light Background
# 1. body_text: set_text_color(200, 200, 200) -> set_text_color(*DARK_GRAY)
content = content.replace("self.set_text_color(200, 200, 200)", "self.set_text_color(*DARK_GRAY)")

# 2. lead_text: set_text_color(240, 240, 240) -> set_text_color(*BLACK)
content = content.replace("self.set_text_color(240, 240, 240)", "self.set_text_color(*BLACK)")

# 3. metric_card:
#    self.set_fill_color(20, 20, 30) -> self.set_fill_color(244, 230, 218) (cream card)
#    self.set_draw_color(40, 40, 50) -> self.set_draw_color(*LIGHT_GRAY)
content = content.replace("self.set_fill_color(20, 20, 30)", "self.set_fill_color(244, 230, 218)")
content = content.replace("self.set_draw_color(40, 40, 50)", "self.set_draw_color(*LIGHT_GRAY)")

# 4. info_box:
#    self.set_fill_color(0, 20, 30) -> self.set_fill_color(226, 236, 245) (pale blue)
#    self.set_draw_color(0, 60, 80) -> self.set_draw_color(156, 193, 231) (sky blue border)
#    self.set_text_color(180, 180, 180) -> self.set_text_color(*DARK_GRAY)
content = content.replace("self.set_fill_color(0, 20, 30)", "self.set_fill_color(226, 236, 245)")
content = content.replace("self.set_draw_color(0, 60, 80)", "self.set_draw_color(156, 193, 231)")
content = content.replace("self.set_text_color(180, 180, 180)", "self.set_text_color(*DARK_GRAY)")

# 5. simple_table:
#    self.set_fill_color(25, 25, 45) -> self.set_fill_color(228, 226, 226) (light gray header)
#    self.set_fill_color(15, 15, 20) if i % 2 == 0 else self.set_fill_color(18, 18, 25)
#    -> self.set_fill_color(255, 255, 255) if i % 2 == 0 else self.set_fill_color(244, 241, 238)
#    self.set_fill_color(0, 30, 40) (highlight row) -> self.set_fill_color(244, 230, 218)
#    self.set_text_color(*WHITE) (highlight text) -> self.set_text_color(*BLACK)
#    self.set_text_color(190, 190, 190) (row text) -> self.set_text_color(*DARK_GRAY)
content = content.replace("self.set_fill_color(25, 25, 45)", "self.set_fill_color(*LIGHT_GRAY)")
content = content.replace("self.set_fill_color(15, 15, 20) if i % 2 == 0 else self.set_fill_color(18, 18, 25)",
                          "self.set_fill_color(255, 255, 255) if i % 2 == 0 else self.set_fill_color(244, 241, 238)")
content = content.replace("self.set_fill_color(0, 30, 40)", "self.set_fill_color(244, 230, 218)")
content = content.replace("self.set_text_color(190, 190, 190)", "self.set_text_color(*DARK_GRAY)")

# 6. Cover page details:
#    pdf.set_fill_color(13, 27, 42) -> pdf.set_fill_color(226, 236, 245) (sky blue background)
#    pdf.set_fill_color(10, 10, 10) (dark left column) -> pdf.set_fill_color(244, 230, 218) (cream column)
content = content.replace("pdf.set_fill_color(13, 27, 42)", "pdf.set_fill_color(226, 236, 245)")
content = content.replace("pdf.set_fill_color(10, 10, 10)", "pdf.set_fill_color(244, 230, 218)")
content = content.replace("pdf.set_text_color(220, 220, 220)", "pdf.set_text_color(*DARK_GRAY)")
content = content.replace("pdf.set_fill_color(15, 15, 15)\n    pdf.rect(0, 270, 210, 27, 'F')",
                          "pdf.set_fill_color(244, 241, 238)\n    pdf.rect(0, 270, 210, 27, 'F')")

# 7. Separator lines color in pages:
#    self.line(20, 14, 190, 14) / self.line(20, 287, 190, 287)
#    Currently draw_color is (40, 40, 40) or (30, 30, 30). Let's replace with LIGHT_GRAY.
content = content.replace("self.set_draw_color(40, 40, 40)", "self.set_draw_color(*LIGHT_GRAY)")
content = content.replace("pdf.set_draw_color(30, 30, 30)", "pdf.set_draw_color(*LIGHT_GRAY)")
content = content.replace("self.set_text_color(50, 50, 50)", "self.set_text_color(*GRAY)")

# 8. Pullquote:
#    pdf.set_fill_color(*CYAN) -> pdf.set_fill_color(*RED) (burnt orange left border)
#    pdf.set_text_color(*CYAN) -> pdf.set_text_color(*RED)
content = content.replace("pdf.set_fill_color(*CYAN)\n    pdf.rect(20, y_start, 2, 20, 'F')",
                          "pdf.set_fill_color(*RED)\n    pdf.rect(20, y_start, 2, 20, 'F')")
content = content.replace("pdf.set_text_color(*CYAN)\n    pdf.set_xy(28, y_start)",
                          "pdf.set_text_color(*RED)\n    pdf.set_xy(28, y_start)")

# 9. CTA boxes:
#    pdf.set_fill_color(*RED) (CTA box bg) -> let's keep RED (burnt orange looks great as CTA bg!)
#    But let's check text colors in CTA box.
#    pdf.set_text_color(*WHITE) -> White is fine on burnt orange background!

# 10. Roadmap timing font color:
#    pdf.set_text_color(*CYAN) -> pdf.set_text_color(*RED)
#    pdf.set_text_color(170, 170, 170) -> pdf.set_text_color(*DARK_GRAY)
content = content.replace("pdf.set_text_color(*CYAN)\n        pdf.set_xy(20, y)\n        pdf.cell(25, 6, time)",
                          "pdf.set_text_color(*RED)\n        pdf.set_xy(20, y)\n        pdf.cell(25, 6, time)")
content = content.replace("pdf.set_text_color(170, 170, 170)", "pdf.set_text_color(*DARK_GRAY)")

# 11. Founder box:
#    pdf.set_fill_color(20, 20, 30) -> pdf.set_fill_color(244, 230, 218)
#    pdf.set_draw_color(50, 50, 60) -> pdf.set_draw_color(*LIGHT_GRAY)
#    pdf.ellipse(28, y + 5, 18, 18, 'F') -> keep CYAN or RED
#    pdf.set_text_color(150, 150, 150) -> pdf.set_text_color(*DARK_GRAY)
content = content.replace("pdf.set_fill_color(20, 20, 30)\n    pdf.set_draw_color(50, 50, 60)\n    pdf.rect(20, y, 170, 28, 'FD')",
                          "pdf.set_fill_color(244, 230, 218)\n    pdf.set_draw_color(*LIGHT_GRAY)\n    pdf.rect(20, y, 170, 28, 'FD')")
content = content.replace("pdf.set_text_color(150, 150, 150)", "pdf.set_text_color(*DARK_GRAY)")

with open(filepath, 'w') as f:
    f.write(content)

print("Magazine layout converted to Dreelio light mode colors successfully.")
