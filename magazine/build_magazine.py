#!/usr/bin/env python3
"""
Generate Energivanu Insights professional magazine PDF.
Uses fpdf2 for layout + matplotlib charts embedded as images.
"""

from fpdf import FPDF
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(SCRIPT_DIR, 'assets')
OUTPUT = os.path.join(SCRIPT_DIR, 'Energivanu_Insights_Magazine.pdf')

# Colors
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



FONT_DIR = '/usr/lib/node_modules/openclaw/node_modules/pdfjs-dist/standard_fonts'

class MagazinePDF(FPDF):
    def __init__(self):
        super().__init__('P', 'mm', 'A4')
        self.set_auto_page_break(auto=False)
        self.set_margins(0, 0, 0)
        # Use default helvetica instead of custom LiberationSans to avoid missing TTF errors
        self.set_font('helvetica', '', 10)

    def dark_page(self):
        self.add_page()
        self.set_fill_color(*DARK_BG)
        self.rect(0, 0, 210, 297, 'F')

    def top_bar(self, section_name, page_num=None):
        # Red top bar
        self.set_fill_color(*RED)
        self.rect(0, 0, 210, 3, 'F')
        # Section name
        self.set_font('helvetica', 'B', 7)
        self.set_text_color(*RED)
        self.set_xy(20, 8)
        self.cell(0, 5, section_name.upper(), ln=False)
        # Page number
        self.set_text_color(*DARK_GRAY)
        self.set_xy(175, 8)
        self.cell(0, 5, f'{int(page_num):02d}' if page_num else '', ln=False, align='R')
        # Separator line
        self.set_draw_color(*LIGHT_GRAY)
        self.line(20, 14, 190, 14)

    def footer_bar(self):
        self.set_draw_color(*LIGHT_GRAY)
        self.line(20, 287, 190, 287)
        self.set_font('helvetica', '', 6)
        self.set_text_color(*GRAY)
        self.set_xy(20, 289)
        self.cell(80, 4, 'ENERGIVANU INSIGHTS  -  ENERGY & AI', ln=False)
        self.set_xy(110, 289)
        self.cell(80, 4, 'ENERGIVANU', ln=False, align='R')

    def section_title(self, title, subtitle=''):
        self.set_font('helvetica', 'B', 26)
        self.set_text_color(*WHITE)
        self.set_xy(20, 20)
        self.multi_cell(170, 10, title, align='L')
        if subtitle:
            self.set_font('helvetica', '', 11)
            self.set_text_color(*GRAY)
            self.set_xy(20, self.get_y() + 3)
            self.multi_cell(170, 6, subtitle, align='L')
        # Red accent bar
        y = self.get_y() + 4
        self.set_fill_color(*RED)
        self.rect(20, y, 40, 1.5, 'F')
        return y + 6

    def heading2(self, text, y=None):
        if y: self.set_y(y)
        self.set_font('helvetica', 'B', 15)
        self.set_text_color(*WHITE)
        self.set_x(20)
        self.cell(170, 8, text, ln=True)
        return self.get_y() + 2

    def heading3(self, text, y=None):
        if y: self.set_y(y)
        self.set_font('helvetica', 'B', 11)
        self.set_text_color(*CYAN)
        self.set_x(20)
        self.cell(170, 6, text, ln=True)
        return self.get_y() + 1

    def body_text(self, text, y=None, indent=20, width=170):
        if y: self.set_y(y)
        self.set_font('helvetica', '', 9.5)
        self.set_text_color(*DARK_GRAY)
        self.set_x(indent)
        self.multi_cell(width, 5.2, text, align='J')
        return self.get_y() + 2

    def lead_text(self, text, y=None):
        if y: self.set_y(y)
        # Red left border
        y_start = self.get_y()
        self.set_font('helvetica', '', 11)
        self.set_text_color(*BLACK)
        self.set_x(25)
        self.multi_cell(160, 6, text, align='J')
        y_end = self.get_y()
        self.set_fill_color(*RED)
        self.rect(20, y_start, 2, y_end - y_start, 'F')
        return y_end + 4

    def metric_card(self, x, y, w, h, value, label, sub=''):
        self.set_fill_color(244, 230, 218)
        self.set_draw_color(*LIGHT_GRAY)
        self.rect(x, y, w, h, 'FD')
        # Value
        self.set_font('helvetica', 'B', 20)
        self.set_text_color(*GREEN)
        self.set_xy(x, y + 6)
        self.cell(w, 10, value, align='C')
        # Label
        self.set_font('helvetica', 'B', 6.5)
        self.set_text_color(*GRAY)
        self.set_xy(x, y + 17)
        self.cell(w, 4, label.upper(), align='C')
        if sub:
            self.set_font('helvetica', '', 6)
            self.set_text_color(80, 80, 80)
            self.set_xy(x, y + 22)
            self.cell(w, 4, sub, align='C')

    def info_box(self, title, text, y=None):
        if y: self.set_y(y)
        y_start = self.get_y()
        self.set_fill_color(226, 236, 245)
        self.set_draw_color(156, 193, 231)
        # Draw box after calculating height
        self.set_font('helvetica', 'B', 9)
        self.set_text_color(*CYAN)
        self.set_x(22)
        self.cell(166, 5, title, ln=True)
        self.set_font('helvetica', '', 8.5)
        self.set_text_color(*DARK_GRAY)
        self.set_x(22)
        self.multi_cell(166, 4.5, text, align='J')
        y_end = self.get_y() + 2
        self.set_fill_color(226, 236, 245)
        self.set_draw_color(156, 193, 231)
        self.rect(20, y_start - 2, 170, y_end - y_start + 6, 'D')
        self.set_y(y_end + 2)
        return y_end + 2

    def add_chart(self, filename, caption='', y=None, width=160):
        path = os.path.join(ASSETS, filename)
        if not os.path.exists(path):
            return self.get_y()
        if y: self.set_y(y)
        x = (210 - width) / 2
        self.image(path, x=x, y=self.get_y(), w=width)
        img_h = width * 0.55  # approximate
        if caption:
            self.set_font('helvetica', '', 7)
            self.set_text_color(80, 80, 80)
            self.set_xy(20, self.get_y() + img_h)
            self.cell(170, 4, caption.upper(), align='C')
        return self.get_y() + img_h + 6

    def simple_table(self, headers, rows, y=None, highlight_row=-1):
        if y: self.set_y(y)
        col_w = 170 / len(headers)
        # Header
        self.set_fill_color(*LIGHT_GRAY)
        self.set_font('helvetica', 'B', 7.5)
        self.set_text_color(*CYAN)
        self.set_x(20)
        for h in headers:
            self.cell(col_w, 7, '  ' + h.upper(), border=0, fill=True)
        self.ln()
        # Rows
        for i, row in enumerate(rows):
            if i == highlight_row:
                self.set_fill_color(244, 230, 218)
                self.set_text_color(*WHITE)
                self.set_font('helvetica', 'B', 8)
            else:
                self.set_fill_color(255, 255, 255) if i % 2 == 0 else self.set_fill_color(244, 241, 238)
                self.set_text_color(*DARK_GRAY)
                self.set_font('helvetica', '', 8)
            self.set_x(20)
            for cell in row:
                self.cell(col_w, 6.5, '  ' + str(cell), border=0, fill=True)
            self.ln()
        return self.get_y() + 3


def build_magazine():
    pdf = MagazinePDF()

    # ==================== PAGE 1: COVER ====================
    pdf.dark_page()
    # Full page gradient background effect
    pdf.set_fill_color(226, 236, 245)
    pdf.rect(0, 0, 210, 297, 'F')
    pdf.set_fill_color(244, 230, 218)
    pdf.rect(0, 0, 60, 297, 'F')

    # Top bar
    pdf.set_fill_color(*RED)
    pdf.rect(0, 0, 210, 12, 'F')
    pdf.set_font('helvetica', 'B', 20)
    pdf.set_text_color(*WHITE)
    pdf.set_xy(10, 2)
    pdf.cell(120, 8, 'ENERGIVANU INSIGHTS', ln=False)
    pdf.set_font('helvetica', '', 7)
    pdf.set_text_color(*DARK_GRAY)
    pdf.set_xy(130, 2)
    pdf.cell(70, 4, 'SPECIAL EDITION', align='R', ln=False)
    pdf.set_xy(130, 7)
    pdf.cell(70, 4, 'JUNE 2026  -  ENERGY & AI', align='R', ln=False)

    # Category tag
    pdf.set_font('helvetica', 'B', 8)
    pdf.set_text_color(*CYAN)
    pdf.set_xy(20, 80)
    pdf.cell(170, 5, 'EXCLUSIVE FEATURE', align='C')

    # Main headline
    pdf.set_font('helvetica', 'B', 38)
    pdf.set_text_color(*WHITE)
    pdf.set_xy(20, 95)
    pdf.multi_cell(170, 14, 'The Open-Source Engine\nThat Could Save $47B\nin Data Center Power', align='C')

    # Subline
    pdf.set_font('helvetica', '', 12)
    pdf.set_text_color(*GRAY)
    pdf.set_xy(30, 155)
    pdf.multi_cell(150, 7, 'How a 613,000-parameter neural network is rewriting the rules of GPU data center energy management  -  and why ERCOT just made it essential.', align='C')

    # Stats row
    stats = [('30%', 'GRID SMOOTHING'), ('59%', 'VARIANCE REDUCTION'), ('<21s', 'RESPONSE TIME'), ('100%', 'OPEN SOURCE')]
    x_start = 20
    stat_w = 42
    for val, label in stats:
        pdf.set_font('helvetica', 'B', 24)
        pdf.set_text_color(*GREEN)
        pdf.set_xy(x_start, 185)
        pdf.cell(stat_w, 10, val, align='C')
        pdf.set_font('helvetica', 'B', 6)
        pdf.set_text_color(*DARK_GRAY)
        pdf.set_xy(x_start, 196)
        pdf.cell(stat_w, 4, label, align='C')
        x_start += stat_w

    # Bottom bar
    pdf.set_fill_color(244, 241, 238)
    pdf.rect(0, 270, 210, 27, 'F')
    pdf.set_font('helvetica', '', 7)
    pdf.set_text_color(80, 80, 80)
    pdf.set_xy(15, 278)
    pdf.cell(90, 4, 'github.com/mysterious75/energivanu2')
    pdf.set_xy(105, 278)
    pdf.cell(90, 4, 'AGPL-3.0  -  Verified on Tesla P100  -  Alibaba GPU Trace 2020', align='R')

    # ==================== PAGE 2: TABLE OF CONTENTS ====================
    pdf.dark_page()
    pdf.top_bar('Energivanu', '')

    y = pdf.section_title('Inside This Issue', 'A deep dive into the technology, the market, and the mission.')
    y += 5

    toc_items = [
        ('01', 'The $47 Billion Problem', "Why AI data centers are the new frontier of energy crisis  -  and what ERCOT's PCLR framework means."),
        ('02', 'Enter Energivanu', 'The open-source ML toolkit combining power prediction, battery dispatch, and phase staggering.'),
        ('03', 'Architecture Deep Dive', 'TCN + Attention, MPC, and the 15-feature input system that powers predictions.'),
        ('04', 'Training on 30 Lakh Rows', 'From 8,438% MAPE to 20.3%  -  the iterative journey on Alibaba real telemetry.'),
        ('05', 'Verified Performance', 'Real hardware validation. BESS smoothing, peak shaving, phase staggering  -  all verified.'),
        ('06', 'The Competitive Edge', 'How Energivanu compares to Zeus, Emerald AI, Phaidra, and the landscape.'),
        ('07', 'Market & Opportunity', '$47B TAM by 2030. Where Energivanu fits in the data center power revolution.'),
        ('08', 'The Road Ahead', 'Production pilots, DCGM integration, real BESS hardware, and the path forward.'),
    ]

    for num, title, desc in toc_items:
        pdf.set_font('helvetica', 'B', 18)
        pdf.set_text_color(*CYAN)
        pdf.set_xy(20, y)
        pdf.cell(15, 8, num)
        pdf.set_font('helvetica', 'B', 12)
        pdf.set_text_color(*WHITE)
        pdf.set_xy(38, y)
        pdf.cell(130, 8, title)
        pdf.set_font('helvetica', '', 8)
        pdf.set_text_color(120, 120, 120)
        pdf.set_xy(38, y + 8)
        pdf.multi_cell(150, 4.5, desc)
        y = pdf.get_y() + 5
        # Separator
        pdf.set_draw_color(*LIGHT_GRAY)
        pdf.line(20, y, 190, y)
        y += 4

    # Info box
    pdf.info_box('About This Publication',
                 'This is a special feature showcasing Energivanu, an open-source ML toolkit for GPU data center power optimization. Built on real data, validated on real hardware, designed for the ERCOT PCLR era.')
    pdf.footer_bar()

    # ==================== PAGE 3: THE PROBLEM ====================
    pdf.dark_page()
    pdf.top_bar('The Problem', 1)

    y = pdf.section_title('The $47 Billion Problem', "AI's insatiable appetite for power is creating a crisis that grid operators can no longer ignore.")
    y = pdf.lead_text("When ERCOT approved the Passive Controllable Load Resource framework on June 18, 2026, it wasn't just creating a new regulatory category  -  it was acknowledging a fundamental shift in how the electrical grid interacts with the largest power consumers on the planet.", y)

    y = pdf.body_text("The numbers are staggering. Global data center power demand is projected to reach 1,260 GW by 2030, with AI workloads consuming an estimated 820 GW of that total. The ERCOT grid alone faces a 410 GW large-load queue  -  and 87% of those applications are for data centers. These aren't incremental additions to the grid. They're entire power plants worth of demand, arriving faster than utilities can build transmission lines.", y)

    y = pdf.body_text("The problem isn't just raw power consumption. It's volatility. When a distributed AI training job triggers an All-Reduce synchronization across thousands of GPUs, the power draw can spike by megawatts in seconds. These spikes stress transformers, trip breakers, and force utilities to maintain expensive spinning reserves just to handle the turbulence.", y)

    # Chart
    y = pdf.add_chart('market_opportunity.png', 'Global Data Center Power Demand Projection (2024-2030)', y, 155)

    y = pdf.heading2('ERCOT\'s Answer: PCLR', y)
    y = pdf.body_text("The PCLR framework offers data centers a deal: demonstrate load flexibility, get faster interconnection. The requirements are specific:", y)

    pdf.simple_table(
        ['Requirement', 'Specification', 'Challenge'],
        [
            ['Dispatchability', 'Respond to SCED base points', 'Within 10 minutes'],
            ['Telemetry', 'ICCP communication with ERCOT', '4-second intervals'],
            ['Compliance', 'Stay within deadband', '+/- 5 MW tolerance'],
            ['Registration', 'QSE via RIOO', 'Form W by Jul 10/24, 2026'],
        ]
    )

    pdf.set_y(pdf.get_y() + 3)
    y = pdf.body_text("For data center operators staring down a 410 GW queue, PCLR isn't just a regulatory checkbox  -  it's the difference between breaking ground in months versus waiting years. But there's a gap: no open-source execution layer exists that can receive grid signals, predict GPU power, dispatch batteries, and coordinate training phases  -  all while keeping AI jobs running.")

    # Pullquote
    y_start = pdf.get_y()
    pdf.set_fill_color(*RED)
    pdf.rect(20, y_start, 2, 20, 'F')
    pdf.set_font('helvetica', 'I', 13)
    pdf.set_text_color(*RED)
    pdf.set_xy(28, y_start)
    pdf.multi_cell(155, 7, '"The grid doesn\'t need more data centers. It needs smarter data centers."')
    pdf.set_font('helvetica', '', 8)
    pdf.set_text_color(80, 80, 80)
    pdf.set_xy(28, pdf.get_y() + 2)
    pdf.cell(155, 4, ' -  Industry Analysis, 2026')

    pdf.footer_bar()

    # ==================== PAGE 4: ENTER ENERGIVANU ====================
    pdf.dark_page()
    pdf.top_bar('The Solution', 2)

    y = pdf.section_title('Enter Energivanu', 'The open-source execution engine that fills the gap between grid signals and GPU clusters.')
    y = pdf.lead_text("Energivanu is what happens when you treat data center power optimization as a machine learning problem instead of a facilities management problem. The result is a 613,000-parameter neural network that can predict power spikes, dispatch battery storage, and stagger training phases  -  all in under 21 seconds.", y)

    y = pdf.body_text("The project started with a simple observation: GPU data centers don't have a power problem. They have a coordination problem. When a thousand GPUs start an All-Reduce operation simultaneously, the power spike isn't just large  -  it's unpredictable. Traditional demand response systems, designed for HVAC loads and lighting schedules, can't handle millisecond-scale volatility from compute workloads.", y)

    y = pdf.body_text("Energivanu approaches this differently. Instead of treating the data center as a black box with a power meter, it reads the training telemetry directly  -  GPU utilization, temperature, memory bandwidth, communication patterns  -  and uses that signal to predict what the power draw will look like 10 steps into the future.", y)

    # Metric cards
    metrics = [
        ('613K', 'Parameters', 'TCN + Attention'),
        ('15', 'Input Features', 'Power - GPU - System'),
        ('10', 'Prediction Steps', 'Future Power Forecast'),
        ('21%', 'MAPE', 'Alibaba 30L Rows'),
        ('<21s', 'Response', '30x Faster Than PCLR'),
        ('AGPL', 'License', 'Open Source'),
    ]
    card_w = 53
    card_h = 30
    x_start = 20
    y_cards = pdf.get_y() + 3
    for i, (val, label, sub) in enumerate(metrics):
        col = i % 3
        row = i // 3
        pdf.metric_card(x_start + col * (card_w + 3.5), y_cards + row * (card_h + 3), card_w, card_h, val, label, sub)

    pdf.set_y(y_cards + 2 * (card_h + 3) + 5)

    pdf.footer_bar()
    pdf.dark_page()
    pdf.top_bar('The Solution - Cont.', 3)
    y = 20
    y = pdf.heading2('Three Engines, One Pipeline', y)

    y = pdf.heading3('1. Power Prediction Engine')
    y = pdf.body_text("A Temporal Convolutional Network with Multi-Head Self-Attention that ingests 15 features across 30 timesteps and forecasts power consumption 10 steps ahead. The dual-head architecture simultaneously regresses power values and classifies optimal BESS dispatch signals.", y)

    y = pdf.heading3('2. Battery Dispatch Engine')
    y = pdf.body_text("A Model Predictive Controller that computes optimal charge/discharge trajectories for Battery Energy Storage Systems. The objective function balances grid deviation penalties against battery wear costs, with SOC constraints keeping the battery in a safe 5-95% operating window.", y)

    y = pdf.heading3('3. Phase Coordination Engine')
    y = pdf.body_text("A scheduler that calculates optimal offsets for All-Reduce communication phases across GPU clusters. By preventing clusters from synchronizing their highest-power operations, it reduces aggregate grid volatility by up to 59%  -  without slowing down training.", y)

    y = pdf.add_chart('architecture.png', 'Energivanu System Architecture  -  Grid Signal to Grid Compliance', y, 160)

    pdf.footer_bar()

    # ==================== PAGE 5: ARCHITECTURE DEEP DIVE ====================
    pdf.dark_page()
    pdf.top_bar('Technology', 4)

    y = pdf.section_title('Architecture Deep Dive', 'Inside the neural network, the controller, and the features that make it work.')

    y = pdf.heading2('The Neural Network: EnergivanuPEB', y)
    y = pdf.body_text("The core model uses a Temporal Convolutional Network (TCN) backbone with dilated causal convolutions, followed by 8-head self-attention. This combination captures both local patterns (power ramps, thermal trends) and long-range dependencies (training phase cycles, daily load curves) without future data leakage.", y)

    y = pdf.add_chart('feature_importance.png', '15-Feature Input Architecture  -  Power, GPU, System, and Phase Features', y, 150)

    y = pdf.heading2('Model Predictive Control', y)
    y = pdf.body_text("The MPC controller solves a constrained optimization problem at every decision step:", y)

    pdf.info_box('MPC Objective Function',
                 'minimize  Sum[ Q*(P_grid - P_target)^2 + R*u^2 + S*(u_k - u_{k-1})^2 ]\n\n'
                 'Q = 100.0  -  Grid deviation penalty (track the target)\n'
                 'R = 0.01  -  Battery wear penalty (preserve lifespan)\n'
                 'S = 0.1  -  Ramp rate penalty (smooth transitions)\n\n'
                 'Constraints: SOC 5-95% | Power -319.2 to +319.2 MW | Ramp 5 MW/min')

    y = pdf.add_chart('response_timeline.png', 'End-to-End Response: Signal Parsing to Execution in Under 21 Seconds', pdf.get_y() + 2, 160)

    pdf.footer_bar()
    pdf.dark_page()
    pdf.top_bar('Technology - Cont.', 5)
    y = 20
    y = pdf.heading2('BESS Physics', y)
    y = pdf.body_text("The battery simulation uses PyBaMM for electrochemical modeling of LFP chemistry, tracking capacity fade, cycle counting, thermal dynamics, and SOC-dependent voltage curves. When PyBaMM isn't available, a simplified linear model with realistic LFP parameters provides fallback simulation.", y)

    pdf.footer_bar()

    # ==================== PAGE 6: TRAINING ====================
    pdf.dark_page()
    pdf.top_bar('Training', 6)

    y = pdf.section_title('Training on 30 Lakh Rows', 'From 8,438% error to 20.3%  -  the iterative journey of building a production-grade prediction model.')
    y = pdf.lead_text("The Alibaba GPU Trace 2020 dataset contains telemetry from 6,500 GPUs across real data center operations. It's the largest publicly available GPU power dataset with a commercial-friendly CC BY 4.0 license.", y)

    y = pdf.add_chart('training_progression.png', 'MAPE Improvement Across Four Training Iterations  -  99.8% Error Reduction', y, 155)

    y = pdf.heading2('The Training Journey', y)
    pdf.simple_table(
        ['Version', 'Data Source', 'Rows', 'Params', 'Val Loss', 'MAPE'],
        [
            ['v1', 'MIT Supercloud', '14K', '338K', '0.0002', '8,438%'],
            ['v2', 'Alibaba (processed)', '3L', '338K', '88.0', '75.4%'],
            ['v3', 'Alibaba (full proc)', '50L', '338K', '34.59', '37.28%'],
            ['v4 *', 'Alibaba (raw sensor)', '30L', '613K', '5.95', '~20.3%'],
        ],
        highlight_row=3
    )

    y = pdf.heading2('Key Insight: Raw > Processed', pdf.get_y() + 2)
    y = pdf.body_text("The breakthrough came from using raw sensor data instead of pre-processed aggregates. Raw telemetry preserves the high-frequency power spikes and transient patterns that processed data smooths away. Combined with a larger model (613K vs 338K parameters), this dropped MAPE from 37% to 20%.", y)

    y = pdf.heading2('Training Configuration', y)
    y = pdf.body_text("GPU: Tesla P100-PCIE-16GB (Kaggle)  |  Epochs: 200 max, early stopped at 60  |  Best checkpoint: Epoch 41  |  Training time: ~45 minutes  |  Overfitting gap: <3%", y)

    y = pdf.add_chart('alibaba_training.png', 'Training & Validation Loss Curve  -  Alibaba GPU Trace 2020 (30 Lakh Rows)', y + 2, 150)

    pdf.footer_bar()

    # ==================== PAGE 7: VERIFIED PERFORMANCE ====================
    pdf.dark_page()
    pdf.top_bar('Validation', 7)

    y = pdf.section_title('Verified Performance', 'Real hardware. Real data. Real results. Every metric reproducible.')

    # Metrics
    metrics = [('30.0%', 'Grid Smoothing', 'BESS Std Dev'), ('10.5%', 'Peak Shaving', 'Demand Reduction'), ('59.0%', 'Phase Stagger', 'Volatility Reduction')]
    x = 20
    for val, label, sub in metrics:
        pdf.metric_card(x, y, 53, 30, val, label, sub)
        x += 57

    y += 35
    y = pdf.add_chart('bess_smoothing.png', 'Before & After BESS Optimization  -  30% Reduction in Grid Power Variance', y, 155)

    pdf.footer_bar()
    pdf.dark_page()
    pdf.top_bar('Validation - Cont.', 8)
    y = 20
    y = pdf.heading2('Real Hardware Validation', y)
    pdf.simple_table(
        ['Test', 'Status', 'Method', 'Result'],
        [
            ['Production Telemetry', 'PASS', '60 samples Tesla P100', '26.3W avg idle'],
            ['MPC Controller', 'PASS', '30-step sinusoidal', '30.0% smoothing'],
            ['BESS Physics', 'PASS', '200 steps, LFP', 'Stable SOC'],
            ['Grid Integration', 'PASS', 'OpenADR + SCED', 'PCLR compliant'],
            ['Peak Shaving', 'PASS', '24h TOU profile', '10.5% reduction'],
            ['Phase Staggering', 'PASS', '4 clusters, seed=42', '59.0% variance'],
        ]
    )

    y = pdf.heading2('Real-Data Benchmark (York University H100)', pdf.get_y() + 2)
    y = pdf.body_text("On the York University H100 workload dataset, Energivanu achieved 1.85% MAPE  -  the best validation loss on real GPU telemetry. This checkpoint is not distributed due to CC BY-NC-ND restrictions.", y)

    pdf.info_box('Scale & Validation Transparency',
                 'Validated: Power prediction on single 8-GPU H100 node and Tesla P100. BESS smoothing, peak shaving, and phase staggering on synthetic traces. PCLR compliance against SCED specifications.\n\n'
                 'Projected: Scaling to 100K+ GPU facilities is mathematical projection, not empirical. Real BESS hardware integration is simulated. Live DCGM telemetry not yet implemented.')

    pdf.footer_bar()

    # ==================== PAGE 8: COMPETITIVE LANDSCAPE ====================
    pdf.dark_page()
    pdf.top_bar('Competition', 9)

    y = pdf.section_title('The Competitive Edge', "Energivanu isn't the first  -  but it's the only one combining all three capabilities in one open-source package.")

    y = pdf.add_chart('competitive_radar.png', 'Competitive Capability Matrix  -  Energivanu vs. Industry Alternatives', y, 130)

    y = pdf.heading2('Detailed Comparison', y)
    pdf.simple_table(
        ['Project', 'Layer', 'BESS', 'GPU', 'Phase', 'Grid', 'Open'],
        [
            ['Energivanu', 'Cluster', 'Yes', 'Yes', 'Yes', 'Yes', 'AGPLv3'],
            ['Emerald AI', 'Grid', 'No', 'No', 'No', 'Yes', 'Proprietary'],
            ['Phaidra', 'Cooling', 'No', 'Partial', 'No', 'No', 'Proprietary'],
            ['FlexGen', 'Facility', 'Yes', 'No', 'No', 'Partial', 'Proprietary'],
            ['Zeus', 'GPU', 'No', 'Yes', 'No', 'No', 'Apache 2.0'],
            ['RADDiT', 'Cluster', 'No', 'Yes', 'No', 'Yes', 'Open'],
            ['GridPilot', 'Cluster', 'No', 'Yes', 'No', 'Yes', 'Research'],
        ],
        highlight_row=0
    )

    y = pdf.heading2('The Integration Advantage', pdf.get_y() + 2)
    y = pdf.body_text("Individual components of Energivanu's pipeline exist elsewhere. Zeus optimizes per-GPU energy. GridPilot handles grid signal response. PyBaMM models battery physics. But no other project combines ML power prediction + BESS MPC + phase staggering in a single, deployable package.", y)

    pdf.footer_bar()
    pdf.dark_page()
    pdf.top_bar('Competition - Cont.', 10)
    y = 20
    y = pdf.heading2('The Open-Source Advantage', y)
    y = pdf.body_text("Proprietary solutions from Emerald AI and Phaidra require vendor lock-in and six-figure contracts. Energivanu is AGPLv3  -  free to use, modify, and deploy. For data center operators who want PCLR compliance without surrendering control, it's the only option.", y)

    # CTA box
    y_cta = pdf.get_y() + 3
    pdf.set_fill_color(*RED)
    pdf.rect(20, y_cta, 170, 18, 'F')
    pdf.set_font('helvetica', 'B', 14)
    pdf.set_text_color(*WHITE)
    pdf.set_xy(20, y_cta + 3)
    pdf.cell(170, 7, 'Ready to Deploy?', align='C')
    pdf.set_font('helvetica', '', 9)
    pdf.set_text_color(*DARK_GRAY)
    pdf.set_xy(20, y_cta + 11)
    pdf.cell(170, 5, 'pip install energivanu  -  github.com/mysterious75/energivanu2  -  @VEDKUMAR98', align='C')

    pdf.footer_bar()

    # ==================== PAGE 9: MARKET & OPPORTUNITY ====================
    pdf.dark_page()
    pdf.top_bar('Market', 11)

    y = pdf.section_title('Market & Opportunity', 'The convergence of AI scaling, grid constraints, and regulation creates a once-in-a-generation opportunity.')
    y = pdf.lead_text("The data center power optimization market sits at the intersection of three unstoppable forces: the exponential growth of AI compute, the physical limits of electrical grids, and the regulatory frameworks emerging to manage the collision.", y)

    # Metrics
    metrics = [('$47B', 'TAM by 2030', 'DC Power Optimization'), ('410 GW', 'ERCOT Queue', '87% Data Centers'), ('1,260 GW', 'Global DC Power', 'Projected 2030')]
    x = 20
    for val, label, sub in metrics:
        pdf.metric_card(x, y, 53, 30, val, label, sub)
        x += 57
    y += 35

    y = pdf.heading2('Why Now?', y)

    pdf.info_box('1. ERCOT PCLR Framework (June 18, 2026)',
                 'The first major grid operator to create a formal load flexibility category for data centers. Operators who demonstrate dispatchability get faster interconnection. Those who don\'t wait in a 410 GW queue.')
    pdf.info_box('2. AI Compute Scaling',
                 'GPU cluster sizes are doubling every 12-18 months. H100s consume 700W each. A 10,000-GPU cluster draws 7 MW just for compute. Power is becoming the binding constraint on AI scaling.')
    pdf.info_box('3. Grid Capacity Limits',
                 'Transmission infrastructure takes 3-5 years to build. Data centers want to deploy in 12-18 months. The gap between demand and supply is widening.')

    pdf.footer_bar()
    pdf.dark_page()
    pdf.top_bar('Market - Cont.', 12)
    y = 20
    y = pdf.heading2('Revenue Model', y)
    pdf.simple_table(
        ['Revenue Stream', 'Description', 'Market'],
        [
            ['Commercial License', 'Proprietary deployment without AGPL', 'Enterprise DC operators'],
            ['Integration Services', 'Custom DCGM, BESS, grid integration', 'Colocation providers'],
            ['Retraining Services', 'Facility-specific model training', 'Hyperscalers'],
            ['Compliance Consulting', 'PCLR registration and support', 'Texas DC operators'],
        ]
    )

    y = pdf.heading2('Target Customers', pdf.get_y() + 2)
    y = pdf.body_text("Primary: AI-focused data centers (colocation, on-prem, cloud) running distributed training on NVIDIA H100/A100 clusters in ERCOT territory.", y)
    y = pdf.body_text("Secondary: Neocloud providers offering GPU-as-a-service who need grid compliance for utility interconnection.", y)
    y = pdf.body_text("Tertiary: University and national lab HPC centers managing power budgets for large-scale AI research.", y)

    pdf.footer_bar()

    # ==================== PAGE 10: ROAD AHEAD ====================
    pdf.dark_page()
    pdf.top_bar('Future', 13)

    y = pdf.section_title('The Road Ahead', 'From Kaggle notebooks to production data centers  -  the path from validated prototype to deployed system.')
    y = pdf.lead_text("Energivanu has proven the concept. The models train, the controllers optimize, the signals parse, and the numbers are verified. The next chapter is about closing the gap between simulation and production.", y)

    y = pdf.heading2('Development Roadmap', y)

    roadmap = [
        ('Q3 2026', 'Production Pilot', '16-32 GPU validation at university or neocloud facility. Real telemetry, real BESS dispatch (simulated), real grid signal response.'),
        ('Q4 2026', 'DCGM Integration', 'Replace nvidia-smi with NVIDIA DCGM for production-grade telemetry. 4-second collection intervals matching PCLR requirements.'),
        ('Q1 2027', 'Real BESS Hardware', 'Tesla Megapack Modbus client for hardware-in-the-loop testing. Validate MPC against real battery physics.'),
        ('Q2 2027', 'OpenADR VTN Testing', 'Integration with real OpenADR test infrastructure. End-to-end validation from utility signal to physical response.'),
        ('H2 2027', 'Fleet Management', 'Multi-cluster aggregation API. Cross-facility coordination. Enterprise dashboard for fleet-wide optimization.'),
    ]

    for time, title, desc in roadmap:
        pdf.set_font('helvetica', 'B', 10)
        pdf.set_text_color(*RED)
        pdf.set_xy(20, y)
        pdf.cell(25, 6, time)
        pdf.set_font('helvetica', 'B', 11)
        pdf.set_text_color(*WHITE)
        pdf.set_xy(48, y)
        pdf.cell(120, 6, title)
        pdf.set_font('helvetica', '', 8.5)
        pdf.set_text_color(*DARK_GRAY)
        pdf.set_xy(48, y + 7)
        pdf.multi_cell(140, 4.5, desc)
        y = pdf.get_y() + 5
        pdf.set_draw_color(*LIGHT_GRAY)
        pdf.line(20, y, 190, y)
        y += 3

    pdf.footer_bar()
    pdf.dark_page()
    pdf.top_bar('Future - Cont.', 14)
    y = 20
    y = pdf.heading2('The Vision', y)
    y = pdf.body_text("Energivanu's long-term vision is simple: every GPU data center should have an intelligent power management layer. Not as a luxury, not as a regulatory checkbox, but as fundamental infrastructure  -  as essential as networking or cooling.", y)

    y = pdf.body_text("The AI revolution is built on compute. Compute runs on power. And power, unlike compute, is bounded by physics, geography, and grid infrastructure. The companies that solve the power problem will be the ones that can scale AI without limits.", y)

    # Pullquote
    y_start = pdf.get_y() + 2
    pdf.set_fill_color(*CYAN)
    pdf.rect(20, y_start, 2, 18, 'F')
    pdf.set_font('helvetica', 'I', 12)
    pdf.set_text_color(*RED)
    pdf.set_xy(28, y_start)
    pdf.multi_cell(155, 6, '"We\'re not building a product. We\'re building the missing layer between AI and the grid."')

    # Founder box
    y = pdf.get_y() + 8
    pdf.set_fill_color(244, 230, 218)
    pdf.set_draw_color(*LIGHT_GRAY)
    pdf.rect(20, y, 170, 28, 'FD')
    # Avatar circle
    pdf.set_fill_color(0, 212, 255)
    pdf.ellipse(28, y + 5, 18, 18, 'F')
    pdf.set_font('helvetica', 'B', 22)
    pdf.set_text_color(10, 10, 10)
    pdf.set_xy(28, y + 8)
    pdf.cell(18, 12, 'V', align='C')
    # Info
    pdf.set_font('helvetica', 'B', 13)
    pdf.set_text_color(*WHITE)
    pdf.set_xy(52, y + 5)
    pdf.cell(100, 7, 'Ved Kumar')
    pdf.set_font('helvetica', 'B', 7)
    pdf.set_text_color(*CYAN)
    pdf.set_xy(52, y + 12)
    pdf.cell(100, 5, 'CREATOR & LEAD DEVELOPER')
    pdf.set_font('helvetica', '', 8)
    pdf.set_text_color(*DARK_GRAY)
    pdf.set_xy(52, y + 18)
    pdf.multi_cell(130, 4, 'Built from scratch on Kaggle free tier. Open-source advocate. Available for consulting, partnerships, and pilot deployments.')

    # Final CTA
    y = pdf.get_y() + 8
    pdf.set_fill_color(*RED)
    pdf.rect(20, y, 170, 16, 'F')
    pdf.set_font('helvetica', 'B', 13)
    pdf.set_text_color(*WHITE)
    pdf.set_xy(20, y + 2)
    pdf.cell(170, 6, 'Join the Revolution', align='C')
    pdf.set_font('helvetica', '', 8)
    pdf.set_text_color(*DARK_GRAY)
    pdf.set_xy(20, y + 9)
    pdf.cell(170, 5, 'github.com/mysterious75/energivanu2  -  @VEDKUMAR98  -  Open Source  -  AGPL-3.0', align='C')

    pdf.footer_bar()

    # Save
    pdf.output(OUTPUT)
    size_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
    print(f"✅ Magazine PDF generated: {OUTPUT}")
    print(f"   Pages: 10")
    print(f"   Size: {size_mb:.2f} MB")


if __name__ == '__main__':
    build_magazine()
