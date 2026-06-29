import re
import os

filepath = 'D:/hacker/energivanu/Energivanu2/magazine/build_magazine.py'
with open(filepath, 'r') as f:
    content = f.read()

# Let's fix the top_bar numbers so they are sequential, starting at 1 for "The Problem"
# Cover has no top_bar.
# TOC has top_bar('Energivanu', 1). Let's remove the number for TOC, or just pass empty string.
# But top_bar expects an integer. Let's make top_bar accept string or int.

top_bar_func_old = """    def top_bar(self, section_name, page_num):
        # Red top bar
        self.set_fill_color(*RED)
        self.rect(0, 0, 210, 3, 'F')
        # Section name
        self.set_font('helvetica', 'B', 7)
        self.set_text_color(*RED)
        self.set_xy(20, 8)
        self.cell(0, 5, section_name.upper(), new_x=XPos.RIGHT, new_y=YPos.TOP)
        # Page number
        self.set_text_color(*DARK_GRAY)
        self.set_xy(175, 8)
        self.cell(0, 5, f'{page_num:02d}', new_x=XPos.RIGHT, new_y=YPos.TOP, align='R')"""

top_bar_func_new = """    def top_bar(self, section_name, page_num=None):
        # Red top bar
        self.set_fill_color(*RED)
        self.rect(0, 0, 210, 3, 'F')
        # Section name
        self.set_font('helvetica', 'B', 7)
        self.set_text_color(*RED)
        self.set_xy(20, 8)
        self.cell(0, 5, section_name.upper(), new_x=XPos.RIGHT, new_y=YPos.TOP)
        # Page number
        if page_num is not None:
            self.set_text_color(*DARK_GRAY)
            self.set_xy(175, 8)
            num_str = f'{page_num:02d}' if isinstance(page_num, int) else str(page_num)
            self.cell(0, 5, num_str, new_x=XPos.RIGHT, new_y=YPos.TOP, align='R')"""

content = content.replace("def top_bar(self, section_name, page_num):", "def top_bar(self, section_name, page_num=None):")
content = content.replace("self.cell(0, 5, f'{page_num:02d}'", "self.cell(0, 5, f'{int(page_num):02d}' if page_num else ''")

# Replace all top_bar calls
# We'll just replace them sequentially.
content = re.sub(r"pdf\.top_bar\('Energivanu', 1\)", "pdf.top_bar('Energivanu', '')", content)
content = re.sub(r"pdf\.top_bar\('The Problem', 2\)", "pdf.top_bar('The Problem', 1)", content)
content = re.sub(r"pdf\.top_bar\('The Solution', 3\)", "pdf.top_bar('The Solution', 2)", content)
content = re.sub(r"pdf\.top_bar\('The Solution - Cont\.', 4\)", "pdf.top_bar('The Solution - Cont.', 3)", content)
content = re.sub(r"pdf\.top_bar\('Technology', 4\)", "pdf.top_bar('Technology', 4)", content)
content = re.sub(r"pdf\.top_bar\('Technology - Cont\.', 5\)", "pdf.top_bar('Technology - Cont.', 5)", content)
content = re.sub(r"pdf\.top_bar\('Training', 5\)", "pdf.top_bar('Training', 6)", content)
content = re.sub(r"pdf\.top_bar\('Validation', 6\)", "pdf.top_bar('Validation', 7)", content)
content = re.sub(r"pdf\.top_bar\('Validation - Cont\.', 7\)", "pdf.top_bar('Validation - Cont.', 8)", content)
content = re.sub(r"pdf\.top_bar\('Competition', 7\)", "pdf.top_bar('Competition', 9)", content)
content = re.sub(r"pdf\.top_bar\('Competition - Cont\.', 8\)", "pdf.top_bar('Competition - Cont.', 10)", content)
content = re.sub(r"pdf\.top_bar\('Market', 8\)", "pdf.top_bar('Market', 11)", content)
content = re.sub(r"pdf\.top_bar\('Market - Cont\.', 9\)", "pdf.top_bar('Market - Cont.', 12)", content)
content = re.sub(r"pdf\.top_bar\('Future', 9\)", "pdf.top_bar('Future', 13)", content)
content = re.sub(r"pdf\.top_bar\('Future - Cont\.', 10\)", "pdf.top_bar('Future - Cont.', 14)", content)

# Now update the TOC to match the new page numbers on the top right
# 01 -> The Problem -> Page 1
# 02 -> Enter Energivanu -> Page 2
# 03 -> Architecture Deep Dive -> Page 4
# 04 -> Training on 30 Lakh Rows -> Page 6
# 05 -> Verified Performance -> Page 7
# 06 -> The Competitive Edge -> Page 9
# 07 -> Market & Opportunity -> Page 11
# 08 -> The Road Ahead -> Page 13

toc_replacement = """    toc_items = [
        ('01', 'The $47 Billion Problem', "Why AI data centers are the new frontier of energy crisis  -  and what ERCOT's PCLR framework means."),
        ('02', 'Enter Energivanu', 'The open-source ML toolkit combining power prediction, battery dispatch, and phase staggering.'),
        ('04', 'Architecture Deep Dive', 'TCN + Attention, MPC, and the 15-feature input system that powers predictions.'),
        ('06', 'Training on 30 Lakh Rows', 'From 8,438% MAPE to 20.3%  -  the iterative journey on Alibaba real telemetry.'),
        ('07', 'Verified Performance', 'Real hardware validation. BESS smoothing, peak shaving, phase staggering  -  all verified.'),
        ('09', 'The Competitive Edge', 'How Energivanu compares to Zeus, Emerald AI, Phaidra, and the landscape.'),
        ('11', 'Market & Opportunity', '$47B TAM by 2030. Where Energivanu fits in the data center power revolution.'),
        ('13', 'The Road Ahead', 'Production pilots, DCGM integration, real BESS hardware, and the path forward.'),
    ]"""

# The TOC numbers on the left are usually "Chapter" numbers: 01, 02, 03, 04. The user specifically asked:
# "isme 03,04,05,06 isss tarah counting chl rhi h , isko 01,02,03,04 aise likho"
# So the TOC numbers (left side) MUST be 01, 02, 03, 04, 05, 06, 07, 08!
# The right side (which doesn't exist in the current script layout) would be the page number.
# Let's keep the TOC numbers exactly as 01, 02, 03, 04, 05, 06, 07, 08 as they already are from my last update.

with open(filepath, 'w') as f:
    f.write(content)
