"""
Enhanced Doodle Recognition Game
================================

An improved machine learning-based drawing recognition game with:
- Smoother animations and transitions
- Better visual feedback
- Enhanced UI/UX
- Real-time confidence visualization
- Improved drawing tools
"""

# Standard library imports
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser
import random, math, time
import numpy as np
from collections import deque, Counter, defaultdict
from PIL import Image, ImageDraw, ImageOps, ImageFilter

# ----------------------------
# Configuration Constants
# ----------------------------
CANVAS_SIZE = 520
PREP_SIZE = 28
DATA_FILE = "doodle_data.npz"
K_NEIGHBORS = 5

ALL_CATEGORIES = ["sun", "cloud", "smiley", "star", "tree", "house"]

DIFF_PRESETS = {
    "Easy":   {"categories": 4, "conf_threshold": 0.25, "smooth_window": 5},
    "Medium": {"categories": 6, "conf_threshold": 0.38, "smooth_window": 7},
    "Hard":   {"categories": 6, "conf_threshold": 0.55, "smooth_window": 9},
}

# Enhanced color scheme with gradients
THEME = {
    "bg": "#0a0e1a",
    "card": "#151b2e",
    "card_hover": "#1a2236",
    "accent": "#6366f1",
    "accent_hover": "#4f46e5",
    "success": "#10b981",
    "danger": "#ef4444",
    "warning": "#f59e0b",
    "text": "#f1f5f9",
    "text_secondary": "#cbd5e1",
    "muted": "#64748b",
    "border": "#1e293b",
    "canvas_bg": "#0f172a",
}

PALETTE = [
    ("White", "#ffffff"),
    ("Yellow", "#fbbf24"),
    ("Red", "#ef4444"),
    ("Green", "#22c55e"),
    ("Blue", "#3b82f6"),
    ("Cyan", "#06b6d4"),
    ("Magenta", "#e879f9"),
    ("Orange", "#f97316"),
]

# ----------------------------
# KNN Classifier
# ----------------------------
class TinyKNN:
    def __init__(self, k=5):
        self.k = k
        self.X = None
        self.y = None
        self.class_counts = {}

    def fit(self, X, y):
        if X is None or len(X) == 0:
            self.X = None; self.y = None; self.class_counts = {}
            return
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=object)
        self._norms = np.linalg.norm(self.X, axis=1) + 1e-8
        self.class_counts = Counter(self.y.tolist())

    def _hybrid_distance(self, x):
        eu = np.linalg.norm(self.X - x, axis=1)
        xnorm = np.linalg.norm(x) + 1e-8
        cos = 1.0 - (np.dot(self.X, x) / (self._norms * xnorm))
        return 0.5 * eu + 0.5 * cos

    def predict_scores(self, x):
        if self.X is None or len(self.X) == 0:
            return {}
        d = self._hybrid_distance(x)
        idx = np.argsort(d)[:min(self.k, len(d))]
        w = 1.0 / (d[idx] + 1e-6)
        scores = {}
        for wi, lab in zip(w, self.y[idx]):
            c = self.class_counts.get(lab, 1)
            scores[lab] = scores.get(lab, 0.0) + wi / math.sqrt(c)
        return scores

    def predict_with_conf(self, x):
        scores = self.predict_scores(x)
        if not scores:
            return None, 0.0
        vals = np.array(list(scores.values()), dtype=np.float32)
        labs = list(scores.keys())
        vals = vals - vals.max()
        exps = np.exp(vals)
        probs = exps / (exps.sum() + 1e-8)
        best_idx = int(np.argmax(probs))
        return labs[best_idx], float(probs[best_idx])

# ----------------------------
# Image Processing
# ----------------------------
def preprocess_image(pil_img):
    img = pil_img.convert("L")
    img = ImageOps.autocontrast(img)
    img = img.resize((PREP_SIZE, PREP_SIZE), Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(0.35))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.reshape(-1)

def draw_seed_shape(label, size=PREP_SIZE):
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    if label == "sun":
        r = size // 4
        cx, cy = size // 2, size // 2
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=200)
        for k in range(10):
            ang = 2 * math.pi * k / 10
            x1 = cx + int(r * 1.2 * math.cos(ang))
            y1 = cy + int(r * 1.2 * math.sin(ang))
            x2 = cx + int(r * 1.9 * math.cos(ang))
            y2 = cy + int(r * 1.9 * math.sin(ang))
            d.line((x1, y1, x2, y2), fill=200, width=2)
    elif label == "cloud":
        centers = [(size*0.32, size*0.55), (size*0.5, size*0.45), (size*0.68, size*0.55)]
        r = size * 0.22
        for (cx, cy) in centers:
            d.ellipse((cx-r, cy-r, cx+r, cy+r), fill=200)
    elif label == "smiley":
        r = size//3
        cx, cy = size//2, size//2
        d.ellipse((cx-r, cy-r, cx+r, cy+r), outline=200, width=2)
        er = max(1, size//30)
        d.ellipse((cx-r//2-er, cy-r//3-er, cx-r//2+er, cy-r//3+er), fill=200)
        d.ellipse((cx+r//2-er, cy-r//3-er, cx+r//2+er, cy+r//3+er), fill=200)
        d.arc((cx-r//2, cy-r//3, cx+r//2, cy+r//2), start=20, end=160, fill=200, width=2)
    elif label == "star":
        R = size*0.36; r = size*0.15; cx, cy = size/2, size/2
        pts = []
        for i in range(10):
            ang = -math.pi/2 + i * math.pi/5
            rad = R if i % 2 == 0 else r
            pts.append((cx + rad*math.cos(ang), cy + rad*math.sin(ang)))
        d.polygon(pts, outline=200)
    elif label == "tree":
        d.rectangle((size*0.45, size*0.6, size*0.55, size*0.92), fill=200)
        d.ellipse((size*0.25, size*0.22, size*0.75, size*0.72), outline=200, width=2)
    elif label == "house":
        d.rectangle((size*0.25, size*0.48, size*0.75, size*0.9), outline=200, width=2)
        d.polygon([(size*0.25, size*0.48), (size*0.5, size*0.15), (size*0.75, size*0.48)], outline=200)
    return np.asarray(img, dtype=np.float32).reshape(-1) / 255.0

def generate_seed_dataset(categories):
    X, y = [], []
    rng = np.random.default_rng(7)
    for lab in categories:
        base = draw_seed_shape(lab, PREP_SIZE).reshape(PREP_SIZE, PREP_SIZE)
        for _ in range(10):
            img = Image.fromarray((base*255).astype(np.uint8), mode="L")
            shift_x = rng.integers(-3, 4)
            shift_y = rng.integers(-3, 4)
            canvas = Image.new("L", (PREP_SIZE+6, PREP_SIZE+6), 0)
            canvas.paste(img, (3+shift_x, 3+shift_y))
            canvas = canvas.resize((PREP_SIZE, PREP_SIZE), Image.BICUBIC)
            arr = np.asarray(canvas, dtype=np.float32) / 255.0
            X.append(arr.reshape(-1)); y.append(lab)
    return np.array(X, dtype=np.float32), np.array(y, dtype=object)

# ----------------------------
# Custom Widgets
# ----------------------------
class RoundedButton(tk.Canvas):
    """Custom button with rounded corners and hover effects"""
    def __init__(self, parent, text="", command=None, style="normal", width=120, height=40, **kwargs):
        super().__init__(parent, width=width, height=height, 
                        bg=THEME["card"], highlightthickness=0, **kwargs)
        
        self.command = command
        self.text = text
        self.style = style
        self.width = width
        self.height = height
        
        # Colors based on style
        if style == "accent":
            self.bg_color = THEME["accent"]
            self.hover_color = THEME["accent_hover"]
            self.text_color = "white"
        elif style == "danger":
            self.bg_color = THEME["danger"]
            self.hover_color = "#dc2626"
            self.text_color = "white"
        elif style == "success":
            self.bg_color = THEME["success"]
            self.hover_color = "#059669"
            self.text_color = "white"
        else:
            self.bg_color = THEME["card_hover"]
            self.hover_color = THEME["border"]
            self.text_color = THEME["text"]
        
        self.current_color = self.bg_color
        self.draw()
        
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)
    
    def draw(self):
        self.delete("all")
        radius = 8
        # Rounded rectangle
        self.create_arc(0, 0, radius*2, radius*2, start=90, extent=90, 
                       fill=self.current_color, outline="")
        self.create_arc(self.width-radius*2, 0, self.width, radius*2, start=0, extent=90,
                       fill=self.current_color, outline="")
        self.create_arc(0, self.height-radius*2, radius*2, self.height, start=180, extent=90,
                       fill=self.current_color, outline="")
        self.create_arc(self.width-radius*2, self.height-radius*2, self.width, self.height,
                       start=270, extent=90, fill=self.current_color, outline="")
        self.create_rectangle(radius, 0, self.width-radius, self.height,
                            fill=self.current_color, outline="")
        self.create_rectangle(0, radius, self.width, self.height-radius,
                            fill=self.current_color, outline="")
        self.create_text(self.width//2, self.height//2, text=self.text,
                        fill=self.text_color, font=("Segoe UI", 10, "bold"))
    
    def on_enter(self, e):
        self.current_color = self.hover_color
        self.draw()
    
    def on_leave(self, e):
        self.current_color = self.bg_color
        self.draw()
    
    def on_click(self, e):
        if self.command:
            self.command()

class AnimatedProgressBar(tk.Canvas):
    """Smooth animated progress bar with gradient effect"""
    def __init__(self, parent, width=240, height=24, **kwargs):
        super().__init__(parent, width=width, height=height,
                        bg=THEME["card"], highlightthickness=0, **kwargs)
        self.width = width
        self.height = height
        self.value = 0
        self.target_value = 0
        self.animating = False
        self.draw_background()
    
    def draw_background(self):
        self.delete("all")
        # Background
        radius = self.height // 2
        self.create_oval(0, 0, self.height, self.height, fill=THEME["border"], outline="")
        self.create_oval(self.width-self.height, 0, self.width, self.height,
                        fill=THEME["border"], outline="")
        self.create_rectangle(radius, 0, self.width-radius, self.height,
                            fill=THEME["border"], outline="")
    
    def set_value(self, value):
        """Set target value (0-100) with smooth animation"""
        self.target_value = max(0, min(100, value))
        if not self.animating:
            self.animate()
    
    def animate(self):
        self.animating = True
        diff = self.target_value - self.value
        if abs(diff) < 0.5:
            self.value = self.target_value
            self.animating = False
        else:
            self.value += diff * 0.15
        
        self.draw_progress()
        
        if self.animating:
            self.after(16, self.animate)
    
    def draw_progress(self):
        self.draw_background()
        if self.value > 0:
            prog_width = (self.width - self.height) * (self.value / 100) + self.height
            radius = self.height // 2
            
            # Gradient effect (simple version with color interpolation)
            color = self._interpolate_color(THEME["accent"], THEME["success"], self.value / 100)
            
            self.create_oval(0, 0, self.height, self.height, fill=color, outline="")
            if prog_width > self.height:
                self.create_oval(prog_width-self.height, 0, prog_width, self.height,
                               fill=color, outline="")
                self.create_rectangle(radius, 0, prog_width-radius, self.height,
                                    fill=color, outline="")
    
    def _interpolate_color(self, color1, color2, t):
        """Interpolate between two hex colors"""
        c1 = [int(color1[i:i+2], 16) for i in (1, 3, 5)]
        c2 = [int(color2[i:i+2], 16) for i in (1, 3, 5)]
        result = [int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3)]
        return f"#{result[0]:02x}{result[1]:02x}{result[2]:02x}"

# ----------------------------
# Main Application
# ----------------------------
class DoodleGameApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Doodle Guess AI")
        self.root.configure(bg=THEME["bg"])
        self._apply_theme()

        # Drawing state
        self.brush_size = 16
        self.brush_color = "#ffffff"
        self.mode = tk.StringVar(value="brush")
        self._last = None
        self._drawing = False

        # Game state
        self.score = 0
        self.rounds_total = 10
        self.round_index = 0
        self.pred_queue = deque(maxlen=7)
        self.diff = "Medium"
        self.active_categories = ALL_CATEGORIES[:]
        self.conf_threshold = DIFF_PRESETS["Medium"]["conf_threshold"]

        # Stats
        self.stats = {
            "started_at": None,
            "guesses": 0,
            "correct": 0,
            "conf_sum_correct": 0.0,
            "per_label": defaultdict(lambda: {"asked": 0, "correct": 0})
        }

        # Achievements & Progress
        self.progress = self._load_progress()
        self.achievements = self._init_achievements()
        self.unlocked_this_session = []

        # Auto-predict state
        self.auto_predict_enabled = tk.BooleanVar(value=False)
        self.auto_predict_job = None

        # Layout
        self.container = tk.Frame(self.root, bg=THEME["bg"])
        self.container.pack(fill="both", expand=True)
        
        self.start_frame = self._build_start_frame(self.container)
        self.game_frame = self._build_game_frame(self.container)
        self.progress_frame = self._build_progress_frame(self.container)
        self._show_frame(self.start_frame)

    def _apply_theme(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure(".", background=THEME["bg"], foreground=THEME["text"],
                   fieldbackground=THEME["card"])
        s.configure("Card.TFrame", background=THEME["card"])
        s.configure("Title.TLabel", font=("Segoe UI", 24, "bold"),
                   background=THEME["bg"], foreground=THEME["text"])
        s.configure("H1.TLabel", font=("Segoe UI", 18, "bold"),
                   background=THEME["card"], foreground=THEME["text"])
        s.configure("H2.TLabel", font=("Segoe UI", 14, "bold"),
                   background=THEME["card"], foreground=THEME["text"])
        s.configure("Body.TLabel", font=("Segoe UI", 11),
                   background=THEME["card"], foreground=THEME["text_secondary"])
        s.configure("Muted.TLabel", font=("Segoe UI", 10),
                   foreground=THEME["muted"], background=THEME["card"])

    def _show_frame(self, frame):
        for child in self.container.winfo_children():
            child.pack_forget()
        frame.pack(fill="both", expand=True)

    def _build_start_frame(self, parent):
        f = tk.Frame(parent, bg=THEME["bg"], padx=40, pady=40)
        
        # Title section
        title_frame = tk.Frame(f, bg=THEME["bg"])
        title_frame.pack(fill="x", pady=(0, 32))
        
        ttk.Label(title_frame, text="🎨 Doodle Guess AI", style="Title.TLabel").pack(anchor="w")
        ttk.Label(title_frame, text="Draw and let AI recognize your doodles in real-time",
                 font=("Segoe UI", 11), foreground=THEME["muted"],
                 background=THEME["bg"]).pack(anchor="w", pady=(4, 0))

        # Settings card
        card = tk.Frame(f, bg=THEME["card"], padx=32, pady=28)
        card.pack(fill="both", expand=True)

        # Difficulty
        diff_frame = tk.Frame(card, bg=THEME["card"])
        diff_frame.pack(fill="x", pady=(0, 24))
        
        ttk.Label(diff_frame, text="Difficulty Level", style="H2.TLabel").pack(anchor="w", pady=(0, 12))
        
        self.diff_var = tk.StringVar(value="Medium")
        diff_buttons = tk.Frame(diff_frame, bg=THEME["card"])
        diff_buttons.pack(fill="x")
        
        for i, (name, desc) in enumerate([
            ("Easy", "4 categories, relaxed"),
            ("Medium", "6 categories, balanced"),
            ("Hard", "6 categories, strict")
        ]):
            btn_frame = tk.Frame(diff_buttons, bg=THEME["card"])
            btn_frame.pack(side="left", padx=(0, 12))
            
            rb = ttk.Radiobutton(btn_frame, text=name, value=name, variable=self.diff_var)
            rb.pack(anchor="w")
            ttk.Label(btn_frame, text=desc, font=("Segoe UI", 9),
                     foreground=THEME["muted"], background=THEME["card"]).pack(anchor="w")

        # Rounds
        rounds_frame = tk.Frame(card, bg=THEME["card"])
        rounds_frame.pack(fill="x", pady=(0, 24))
        
        ttk.Label(rounds_frame, text="Number of Rounds", style="H2.TLabel").pack(anchor="w", pady=(0, 8))
        
        self.rounds_var = tk.IntVar(value=10)
        rounds_scale = ttk.Scale(rounds_frame, from_=3, to=25, orient="horizontal",
                                variable=self.rounds_var, length=300)
        rounds_scale.pack(anchor="w", pady=(0, 4))
        
        self.rounds_label = ttk.Label(rounds_frame, text="10 rounds",
                                     font=("Segoe UI", 10), foreground=THEME["text_secondary"],
                                     background=THEME["card"])
        self.rounds_label.pack(anchor="w")
        
        def update_rounds_label(val):
            self.rounds_label.config(text=f"{int(float(val))} rounds")
        rounds_scale.configure(command=update_rounds_label)

        # Info
        info_frame = tk.Frame(card, bg=THEME["border"], padx=16, pady=12)
        info_frame.pack(fill="x", pady=(0, 24))
        
        ttk.Label(info_frame, text="💡 Tips:",
                 font=("Segoe UI", 10, "bold"), foreground=THEME["warning"],
                 background=THEME["border"]).pack(anchor="w")
        ttk.Label(info_frame,
                 text="• Draw clearly in the center of the canvas\n• Use the 'Teach' button if AI makes mistakes\n• Press Space to make quick predictions",
                 font=("Segoe UI", 9), foreground=THEME["text_secondary"],
                 background=THEME["border"], justify="left").pack(anchor="w", pady=(4, 0))

        # Start button
        btn_container = tk.Frame(card, bg=THEME["card"])
        btn_container.pack(anchor="w", fill="x")
        
        start_btn = RoundedButton(btn_container, text="Start Game", style="accent",
                                 width=160, height=48, command=self._start_game)
        start_btn.pack(side="left", padx=(0, 12))
        
        progress_btn = RoundedButton(btn_container, text="📊 Progress", style="normal",
                                    width=140, height=48, command=self._show_progress)
        progress_btn.pack(side="left")

        return f

    def _build_game_frame(self, parent):
        f = tk.Frame(parent, bg=THEME["bg"], padx=16, pady=16)
        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=0)
        f.rowconfigure(1, weight=1)

        # Header
        header = tk.Frame(f, bg=THEME["card"], padx=20, pady=16)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        header.columnconfigure(1, weight=1)
        
        ttk.Label(header, text="🎨 Doodle Guess AI", style="H1.TLabel").grid(row=0, column=0, sticky="w")
        
        score_frame = tk.Frame(header, bg=THEME["card"])
        score_frame.grid(row=0, column=2, sticky="e")
        ttk.Label(score_frame, text="Score", font=("Segoe UI", 9),
                 foreground=THEME["muted"], background=THEME["card"]).pack()
        self.lbl_score = ttk.Label(score_frame, text="0", font=("Segoe UI", 20, "bold"),
                                  foreground=THEME["success"], background=THEME["card"])
        self.lbl_score.pack()

        # Left: Canvas area
        left = tk.Frame(f, bg=THEME["bg"])
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        # Toolbar
        toolbar = tk.Frame(left, bg=THEME["card"], padx=16, pady=12)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        toolbar.columnconfigure(6, weight=1)

        # Tool buttons
        self.btn_brush = RoundedButton(toolbar, text="✏️ Brush", width=90, height=36,
                                      style="accent", command=lambda: self._set_mode("brush"))
        self.btn_brush.grid(row=0, column=0, padx=(0, 8))
        
        self.btn_eraser = RoundedButton(toolbar, text="🧹 Eraser", width=90, height=36,
                                       command=lambda: self._set_mode("eraser"))
        self.btn_eraser.grid(row=0, column=1, padx=(0, 16))

        # Size control
        ttk.Label(toolbar, text="Size", font=("Segoe UI", 9),
                 foreground=THEME["muted"], background=THEME["card"]).grid(row=0, column=2, padx=(0, 8))
        
        self.size_var = tk.IntVar(value=self.brush_size)
        size_scale = ttk.Scale(toolbar, from_=4, to=48, orient="horizontal",
                              command=self._on_size_change, variable=self.size_var, length=120)
        size_scale.grid(row=0, column=3, padx=(0, 16))

        # Color palette
        color_frame = tk.Frame(toolbar, bg=THEME["card"])
        color_frame.grid(row=0, column=4, padx=(0, 16))
        for i, (name, hexv) in enumerate(PALETTE[:6]):
            b = tk.Canvas(color_frame, width=24, height=24, bg=THEME["card"],
                         highlightthickness=0)
            b.grid(row=0, column=i, padx=2)
            b.create_oval(4, 4, 20, 20, fill=hexv, outline=THEME["border"], width=1)
            b.bind("<Button-1>", lambda e, hv=hexv: self._set_color(hv))

        # Action buttons
        RoundedButton(toolbar, text="🗑️ Clear", width=80, height=36,
                     command=self.clear_canvas).grid(row=0, column=7, sticky="e", padx=(0, 8))
        
        RoundedButton(toolbar, text="🔍 Guess", width=80, height=36, style="success",
                     command=self.on_guess).grid(row=0, column=8, sticky="e")

        # Canvas card
        canvas_card = tk.Frame(left, bg=THEME["card"], padx=12, pady=12)
        canvas_card.grid(row=1, column=0, sticky="nsew")
        
        canvas_border = tk.Frame(canvas_card, bg=THEME["border"], padx=2, pady=2)
        canvas_border.pack(fill="both", expand=True)
        
        self.canvas = tk.Canvas(canvas_border, width=CANVAS_SIZE, height=CANVAS_SIZE,
                               bg=THEME["canvas_bg"], bd=0, highlightthickness=0,
                               cursor="circle")
        self.canvas.pack()

        # PIL canvas
        self.pil_canvas = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_canvas)

        self.canvas.bind("<ButtonPress-1>", self._on_draw_start)
        self.canvas.bind("<B1-Motion>", self._on_draw_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_draw_end)

        # Right sidebar
        sidebar = tk.Frame(f, bg=THEME["card"], padx=20, pady=20, width=320)
        sidebar.grid(row=1, column=1, sticky="ns")
        sidebar.grid_propagate(False)

        # Target section
        target_card = tk.Frame(sidebar, bg=THEME["border"], padx=16, pady=16)
        target_card.pack(fill="x", pady=(0, 16))
        
        ttk.Label(target_card, text="🎯 Draw This",
                 font=("Segoe UI", 11, "bold"), foreground=THEME["warning"],
                 background=THEME["border"]).pack(anchor="w")
        
        self.lbl_target = ttk.Label(target_card, text="—",
                                    font=("Segoe UI", 22, "bold"),
                                    foreground=THEME["text"], background=THEME["border"])
        self.lbl_target.pack(anchor="w", pady=(4, 0))

        # Prediction section
        pred_card = tk.Frame(sidebar, bg=THEME["card_hover"], padx=16, pady=16)
        pred_card.pack(fill="x", pady=(0, 16))
        
        ttk.Label(pred_card, text="AI Prediction",
                 font=("Segoe UI", 10), foreground=THEME["muted"],
                 background=THEME["card_hover"]).pack(anchor="w")
        
        self.lbl_pred = ttk.Label(pred_card, text="—",
                                 font=("Segoe UI", 18, "bold"),
                                 foreground=THEME["accent"],
                                 background=THEME["card_hover"])
        self.lbl_pred.pack(anchor="w", pady=(4, 12))

        # Animated confidence bar
        self.conf_bar = AnimatedProgressBar(pred_card, width=260, height=20)
        self.conf_bar.pack(fill="x", pady=(0, 8))
        
        self.lbl_conf = ttk.Label(pred_card, text="Confidence: —",
                                 font=("Segoe UI", 10), foreground=THEME["text_secondary"],
                                 background=THEME["card_hover"])
        self.lbl_conf.pack(anchor="w")

        # Auto-predict toggle
        auto_frame = tk.Frame(sidebar, bg=THEME["card"])
        auto_frame.pack(fill="x", pady=(0, 16))
        
        check = ttk.Checkbutton(auto_frame, text="🤖 Auto-predict while drawing",
                               variable=self.auto_predict_enabled,
                               command=self._toggle_auto_predict)
        check.pack(anchor="w")

        # Actions
        ttk.Label(sidebar, text="Actions", font=("Segoe UI", 11, "bold"),
                 foreground=THEME["text"], background=THEME["card"]).pack(anchor="w", pady=(0, 8))
        
        RoundedButton(sidebar, text="📚 Teach as Target", width=260, height=40,
                     command=self.on_teach).pack(pady=(0, 8))
        
        RoundedButton(sidebar, text="🔄 New Target", width=260, height=40,
                     command=self.next_target).pack(pady=(0, 16))

        # Data management
        ttk.Label(sidebar, text="Dataset", font=("Segoe UI", 11, "bold"),
                 foreground=THEME["text"], background=THEME["card"]).pack(anchor="w", pady=(8, 8))
        
        RoundedButton(sidebar, text="💾 Save Dataset", width=260, height=36,
                     command=self.save_dataset).pack(pady=(0, 6))
        
        RoundedButton(sidebar, text="📁 Load Dataset", width=260, height=36,
                     command=self.load_dataset).pack(pady=(0, 16))

        # Exit
        RoundedButton(sidebar, text="⬅️ Back to Menu", width=260, height=36,
                     style="danger", command=self._exit_to_start).pack(pady=(16, 0))

        # Status bar
        status = tk.Frame(f, bg=THEME["card"], padx=16, pady=10)
        status.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        
        ttk.Label(status,
                 text="⌨️ Space: Guess  •  T: Teach  •  N: New Target  •  C: Clear  •  B: Brush  •  E: Eraser",
                 font=("Segoe UI", 9), foreground=THEME["muted"],
                 background=THEME["card"]).pack()

        self._bind_shortcuts()
        return f

    def _build_progress_frame(self, parent):
        """Build achievements and progress tracking screen"""
        f = tk.Frame(parent, bg=THEME["bg"], padx=40, pady=40)
        
        # Header
        header = tk.Frame(f, bg=THEME["bg"])
        header.pack(fill="x", pady=(0, 24))
        
        ttk.Label(header, text="📊 Your Progress", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="Track your achievements and statistics",
                 font=("Segoe UI", 11), foreground=THEME["muted"],
                 background=THEME["bg"]).pack(anchor="w", pady=(4, 0))

        # Main content area
        content = tk.Frame(f, bg=THEME["bg"])
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=2)
        content.columnconfigure(1, weight=1)

        # Left: Statistics
        stats_card = tk.Frame(content, bg=THEME["card"], padx=24, pady=20)
        stats_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        ttk.Label(stats_card, text="🎮 Game Statistics", style="H1.TLabel").pack(anchor="w", pady=(0, 16))

        # Stats grid
        stats_grid = tk.Frame(stats_card, bg=THEME["card"])
        stats_grid.pack(fill="x", pady=(0, 20))

        stat_items = [
            ("Total Games", self.progress['games_played']),
            ("Total Rounds Won", self.progress['rounds_won']),
            ("Total Guesses", self.progress['total_guesses']),
            ("Correct Guesses", self.progress['correct_guesses']),
            ("Drawings Taught", self.progress['drawings_taught']),
            ("Best Streak", self.progress['best_streak']),
        ]

        for i, (label, value) in enumerate(stat_items):
            row = i // 2
            col = i % 2
            
            stat_frame = tk.Frame(stats_grid, bg=THEME["card_hover"], padx=16, pady=12)
            stat_frame.grid(row=row, column=col, sticky="ew", padx=(0, 8) if col == 0 else (0, 0), pady=(0, 8))
            
            tk.Label(stat_frame, text=str(value), font=("Segoe UI", 24, "bold"),
                    fg=THEME["accent"], bg=THEME["card_hover"]).pack(anchor="w")
            tk.Label(stat_frame, text=label, font=("Segoe UI", 10),
                    fg=THEME["muted"], bg=THEME["card_hover"]).pack(anchor="w")

        stats_grid.columnconfigure(0, weight=1)
        stats_grid.columnconfigure(1, weight=1)

        # Accuracy stats
        if self.progress['total_guesses'] > 0:
            accuracy = (self.progress['correct_guesses'] / self.progress['total_guesses']) * 100
        else:
            accuracy = 0

        ttk.Label(stats_card, text="Overall Accuracy", style="H2.TLabel").pack(anchor="w", pady=(8, 8))
        
        acc_bar = AnimatedProgressBar(stats_card, width=400, height=32)
        acc_bar.pack(fill="x", pady=(0, 8))
        acc_bar.set_value(accuracy)
        
        tk.Label(stats_card, text=f"{accuracy:.1f}%", font=("Segoe UI", 18, "bold"),
                fg=THEME["success"], bg=THEME["card"]).pack(anchor="w")

        # Per-category performance
        ttk.Label(stats_card, text="Category Performance", style="H2.TLabel").pack(anchor="w", pady=(20, 12))
        
        cat_frame = tk.Frame(stats_card, bg=THEME["card"])
        cat_frame.pack(fill="x")

        for cat in ALL_CATEGORIES:
            cat_stats = self.progress['per_category'].get(cat, {'asked': 0, 'correct': 0})
            asked = cat_stats['asked']
            correct = cat_stats['correct']
            acc = (correct / asked * 100) if asked > 0 else 0

            cat_row = tk.Frame(cat_frame, bg=THEME["card_hover"], padx=12, pady=8)
            cat_row.pack(fill="x", pady=(0, 4))

            tk.Label(cat_row, text=cat.title(), font=("Segoe UI", 11),
                    fg=THEME["text"], bg=THEME["card_hover"], width=10, anchor="w").pack(side="left")
            
            bar_container = tk.Frame(cat_row, bg=THEME["card_hover"])
            bar_container.pack(side="left", fill="x", expand=True, padx=(8, 8))
            
            mini_bar = AnimatedProgressBar(bar_container, width=200, height=16)
            mini_bar.pack(side="left")
            mini_bar.set_value(acc)

            tk.Label(cat_row, text=f"{correct}/{asked}", font=("Segoe UI", 10),
                    fg=THEME["muted"], bg=THEME["card_hover"]).pack(side="right")

        # Right: Achievements
        achieve_card = tk.Frame(content, bg=THEME["card"], padx=24, pady=20)
        achieve_card.grid(row=0, column=1, sticky="nsew")

        ttk.Label(achieve_card, text="🏆 Achievements", style="H1.TLabel").pack(anchor="w", pady=(0, 16))

        # Scrollable achievements list
        achieve_canvas = tk.Canvas(achieve_card, bg=THEME["card"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(achieve_card, orient="vertical", command=achieve_canvas.yview)
        achieve_list = tk.Frame(achieve_canvas, bg=THEME["card"])

        achieve_list.bind("<Configure>", lambda e: achieve_canvas.configure(scrollregion=achieve_canvas.bbox("all")))
        achieve_canvas.create_window((0, 0), window=achieve_list, anchor="nw")
        achieve_canvas.configure(yscrollcommand=scrollbar.set)

        # Display achievements
        for ach_id, achievement in self.achievements.items():
            unlocked = achievement['unlocked']
            
            ach_frame = tk.Frame(achieve_list, 
                                bg=THEME["card_hover"] if unlocked else THEME["border"],
                                padx=12, pady=12)
            ach_frame.pack(fill="x", pady=(0, 8))

            # Icon and title
            icon_label = tk.Label(ach_frame, text=achievement['icon'], font=("Segoe UI", 20),
                                 bg=ach_frame['bg'])
            icon_label.pack(anchor="w")

            title_label = tk.Label(ach_frame, text=achievement['title'],
                                  font=("Segoe UI", 11, "bold"),
                                  fg=THEME["text"] if unlocked else THEME["muted"],
                                  bg=ach_frame['bg'], anchor="w")
            title_label.pack(anchor="w", fill="x")

            desc_label = tk.Label(ach_frame, text=achievement['description'],
                                 font=("Segoe UI", 9),
                                 fg=THEME["text_secondary"] if unlocked else THEME["muted"],
                                 bg=ach_frame['bg'], anchor="w", wraplength=250, justify="left")
            desc_label.pack(anchor="w", fill="x", pady=(4, 0))

            # Progress for locked achievements
            if not unlocked and 'progress' in achievement:
                prog_text = f"{achievement['progress']}/{achievement['target']}"
                prog_label = tk.Label(ach_frame, text=prog_text,
                                     font=("Segoe UI", 9, "italic"),
                                     fg=THEME["muted"], bg=ach_frame['bg'])
                prog_label.pack(anchor="w", pady=(4, 0))

        achieve_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Back button
        back_btn = RoundedButton(f, text="⬅️ Back to Menu", width=160, height=44,
                                command=lambda: self._show_frame(self.start_frame))
        back_btn.pack(pady=(20, 0))

        return f

    # -------------- Game Logic --------------
    def _init_achievements(self):
        """Initialize achievement definitions"""
        return {
            'first_win': {
                'title': 'First Victory',
                'description': 'Win your first round',
                'icon': '🎯',
                'unlocked': self.progress['rounds_won'] >= 1
            },
            'perfect_game': {
                'title': 'Perfect Score',
                'description': 'Complete a game with 100% accuracy',
                'icon': '💯',
                'unlocked': self.progress.get('perfect_games', 0) >= 1
            },
            'ten_wins': {
                'title': 'Seasoned Artist',
                'description': 'Win 10 rounds',
                'icon': '🎨',
                'unlocked': self.progress['rounds_won'] >= 10,
                'progress': min(self.progress['rounds_won'], 10),
                'target': 10
            },
            'fifty_wins': {
                'title': 'Master Doodler',
                'description': 'Win 50 rounds',
                'icon': '👑',
                'unlocked': self.progress['rounds_won'] >= 50,
                'progress': min(self.progress['rounds_won'], 50),
                'target': 50
            },
            'teacher': {
                'title': 'AI Teacher',
                'description': 'Teach the AI 20 drawings',
                'icon': '👨‍🏫',
                'unlocked': self.progress['drawings_taught'] >= 20,
                'progress': min(self.progress['drawings_taught'], 20),
                'target': 20
            },
            'streak_5': {
                'title': 'Hot Streak',
                'description': 'Get 5 correct guesses in a row',
                'icon': '🔥',
                'unlocked': self.progress['best_streak'] >= 5,
                'progress': min(self.progress['best_streak'], 5),
                'target': 5
            },
            'speed_demon': {
                'title': 'Speed Demon',
                'description': 'Win a round in under 60 seconds',
                'icon': '⚡',
                'unlocked': self.progress.get('fastest_game', 999) < 60
            },
            'all_categories': {
                'title': 'Category Master',
                'description': 'Win at least once in every category',
                'icon': '🌟',
                'unlocked': all(self.progress['per_category'].get(cat, {}).get('correct', 0) > 0 
                               for cat in ALL_CATEGORIES)
            },
            'hundred_games': {
                'title': 'Dedicated Player',
                'description': 'Play 100 games',
                'icon': '💪',
                'unlocked': self.progress['games_played'] >= 100,
                'progress': min(self.progress['games_played'], 100),
                'target': 100
            },
            'high_confidence': {
                'title': 'Confidence King',
                'description': 'Win with 95%+ confidence',
                'icon': '😎',
                'unlocked': self.progress.get('high_conf_wins', 0) >= 1
            }
        }

    def _load_progress(self):
        """Load progress from file or create new"""
        try:
            import json
            with open('doodle_progress.json', 'r') as f:
                return json.load(f)
        except:
            return {
                'games_played': 0,
                'rounds_won': 0,
                'total_guesses': 0,
                'correct_guesses': 0,
                'drawings_taught': 0,
                'best_streak': 0,
                'current_streak': 0,
                'perfect_games': 0,
                'fastest_game': 999,
                'high_conf_wins': 0,
                'per_category': {cat: {'asked': 0, 'correct': 0} for cat in ALL_CATEGORIES}
            }

    def _save_progress(self):
        """Save progress to file"""
        try:
            import json
            with open('doodle_progress.json', 'w') as f:
                json.dump(self.progress, f, indent=2)
        except Exception as e:
            print(f"Failed to save progress: {e}")

    def _check_achievements(self):
        """Check for newly unlocked achievements"""
        newly_unlocked = []
        
        for ach_id, achievement in self.achievements.items():
            was_unlocked = achievement['unlocked']
            
            # Re-evaluate unlock condition
            if ach_id == 'first_win':
                achievement['unlocked'] = self.progress['rounds_won'] >= 1
            elif ach_id == 'perfect_game':
                achievement['unlocked'] = self.progress.get('perfect_games', 0) >= 1
            elif ach_id == 'ten_wins':
                achievement['unlocked'] = self.progress['rounds_won'] >= 10
                achievement['progress'] = min(self.progress['rounds_won'], 10)
            elif ach_id == 'fifty_wins':
                achievement['unlocked'] = self.progress['rounds_won'] >= 50
                achievement['progress'] = min(self.progress['rounds_won'], 50)
            elif ach_id == 'teacher':
                achievement['unlocked'] = self.progress['drawings_taught'] >= 20
                achievement['progress'] = min(self.progress['drawings_taught'], 20)
            elif ach_id == 'streak_5':
                achievement['unlocked'] = self.progress['best_streak'] >= 5
                achievement['progress'] = min(self.progress['best_streak'], 5)
            elif ach_id == 'speed_demon':
                achievement['unlocked'] = self.progress.get('fastest_game', 999) < 60
            elif ach_id == 'all_categories':
                achievement['unlocked'] = all(self.progress['per_category'].get(cat, {}).get('correct', 0) > 0 
                                             for cat in ALL_CATEGORIES)
            elif ach_id == 'hundred_games':
                achievement['unlocked'] = self.progress['games_played'] >= 100
                achievement['progress'] = min(self.progress['games_played'], 100)
            elif ach_id == 'high_confidence':
                achievement['unlocked'] = self.progress.get('high_conf_wins', 0) >= 1
            
            # Track newly unlocked
            if achievement['unlocked'] and not was_unlocked:
                newly_unlocked.append((ach_id, achievement))
        
        # Show notification for new achievements
        for ach_id, achievement in newly_unlocked:
            self._show_achievement_unlock(achievement)
            self.unlocked_this_session.append(ach_id)

    def _show_achievement_unlock(self, achievement):
        """Show achievement unlock notification"""
        notif = tk.Toplevel(self.root)
        notif.title("Achievement Unlocked!")
        notif.configure(bg=THEME["card"])
        notif.transient(self.root)
        notif.attributes('-topmost', True)
        
        # Center on screen
        notif.geometry("350x150")
        notif.update_idletasks()
        x = (notif.winfo_screenwidth() // 2) - (350 // 2)
        y = (notif.winfo_screenheight() // 2) - (150 // 2)
        notif.geometry(f"350x150+{x}+{y}")

        container = tk.Frame(notif, bg=THEME["success"], padx=24, pady=20)
        container.pack(fill="both", expand=True)

        tk.Label(container, text="🎉 Achievement Unlocked!",
                font=("Segoe UI", 14, "bold"), fg="white", bg=THEME["success"]).pack()

        tk.Label(container, text=f"{achievement['icon']} {achievement['title']}",
                font=("Segoe UI", 16, "bold"), fg="white", bg=THEME["success"]).pack(pady=(8, 4))

        tk.Label(container, text=achievement['description'],
                font=("Segoe UI", 10), fg="white", bg=THEME["success"]).pack()

        # Auto close after 3 seconds
        notif.after(3000, notif.destroy)

    def _show_progress(self):
        """Show progress screen"""
        # Refresh achievements before showing
        self.achievements = self._init_achievements()
        self._show_frame(self.progress_frame)
        
        # Rebuild the frame to update stats
        self.progress_frame.destroy()
        self.progress_frame = self._build_progress_frame(self.container)
        self._show_frame(self.progress_frame)

    def _start_game(self):
        self.diff = self.diff_var.get()
        preset = DIFF_PRESETS[self.diff]
        self.rounds_total = max(3, int(self.rounds_var.get()))
        
        rng = random.Random(42)
        count = preset["categories"]
        chosen = rng.sample(ALL_CATEGORIES, k=count) if count < len(ALL_CATEGORIES) else ALL_CATEGORIES[:]
        self.active_categories = chosen
        self.conf_threshold = preset["conf_threshold"]
        self.pred_queue = deque(maxlen=preset["smooth_window"])

        # Initialize model
        self.model = TinyKNN(k=K_NEIGHBORS)
        X, y = generate_seed_dataset(self.active_categories)
        self.X, self.y = X, y
        self.model.fit(self.X, self.y)

        # Reset game state
        self.score = 0
        self.round_index = 0
        self.lbl_score.config(text="0")
        self.unlocked_this_session = []
        self.stats = {
            "started_at": time.time(),
            "guesses": 0,
            "correct": 0,
            "conf_sum_correct": 0.0,
            "per_label": defaultdict(lambda: {"asked": 0, "correct": 0})
        }

        self._show_frame(self.game_frame)
        self.next_target(auto_clear=True)

    def _exit_to_start(self):
        if self.auto_predict_job:
            self.root.after_cancel(self.auto_predict_job)
            self.auto_predict_job = None
        self._show_frame(self.start_frame)

    # -------------- Drawing --------------
    def _on_draw_start(self, event):
        self._last = (event.x, event.y)
        self._drawing = True
        self._draw_point(event.x, event.y)

    def _on_draw_move(self, event):
        if self._last is None or not self._drawing:
            return
        
        x0, y0 = self._last
        x1, y1 = event.x, event.y
        
        if self.mode.get() == "eraser":
            color_canvas = THEME["canvas_bg"]
            color_pil = 0
        else:
            color_canvas = self.brush_color
            color_pil = 255
        
        self.canvas.create_line(x0, y0, x1, y1, fill=color_canvas,
                               width=self.brush_size, capstyle=tk.ROUND, smooth=True)
        self.pil_draw.line((x0, y0, x1, y1), fill=color_pil, width=self.brush_size)
        self._last = (x1, y1)

    def _on_draw_end(self, event):
        self._last = None
        self._drawing = False

    def _draw_point(self, x, y):
        r = self.brush_size // 2
        if self.mode.get() == "eraser":
            color_canvas = THEME["canvas_bg"]
            color_pil = 0
        else:
            color_canvas = self.brush_color
            color_pil = 255
        
        self.canvas.create_oval(x - r, y - r, x + r, y + r,
                               fill=color_canvas, outline=color_canvas)
        self.pil_draw.ellipse((x - r, y - r, x + r, y + r), fill=color_pil)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.pil_canvas = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_canvas)
        self.lbl_pred.config(text="—")
        self.lbl_conf.config(text="Confidence: —")
        self.conf_bar.set_value(0)
        self.pred_queue.clear()

    def _on_size_change(self, val):
        try:
            self.brush_size = int(float(val))
        except:
            pass

    def _set_color(self, hexv):
        self.brush_color = hexv
        self._set_mode("brush")

    def _set_mode(self, mode):
        self.mode.set(mode)
        if mode == "brush":
            self.btn_brush.style = "accent"
            self.btn_brush.draw()
            self.btn_eraser.style = "normal"
            self.btn_eraser.draw()
        else:
            self.btn_eraser.style = "accent"
            self.btn_eraser.draw()
            self.btn_brush.style = "normal"
            self.btn_brush.draw()

    # -------------- Auto-predict --------------
    def _toggle_auto_predict(self):
        if self.auto_predict_enabled.get():
            self._schedule_auto_predict()
        else:
            if self.auto_predict_job:
                self.root.after_cancel(self.auto_predict_job)
                self.auto_predict_job = None

    def _schedule_auto_predict(self):
        if self.auto_predict_enabled.get():
            self._auto_predict_silent()
            self.auto_predict_job = self.root.after(500, self._schedule_auto_predict)

    def _auto_predict_silent(self):
        """Run prediction without advancing round"""
        x = self._capture_vector()
        pred, conf = self.model.predict_with_conf(x)
        
        if pred is None:
            return
        
        pred_s, conf_s = self._smoothed_prediction(pred, conf)
        self.lbl_pred.config(text=pred_s)
        self.lbl_conf.config(text=f"Confidence: {int(conf_s*100)}%")
        self.conf_bar.set_value(int(conf_s * 100))

    # -------------- Model / Prediction --------------
    def _capture_vector(self):
        img = self.pil_canvas.copy()
        arr = np.array(img)
        ys, xs = np.where(arr > 10)
        
        if len(xs) > 0 and len(ys) > 0:
            minx, maxx = xs.min(), xs.max()
            miny, maxy = ys.min(), ys.max()
            minx = max(0, minx - 12)
            miny = max(0, miny - 12)
            maxx = min(arr.shape[1]-1, maxx + 12)
            maxy = min(arr.shape[0]-1, maxy + 12)
            img = img.crop((minx, miny, maxx, maxy))
        
        w, h = img.size
        side = max(w, h)
        sq = Image.new("L", (side, side), 0)
        sq.paste(img, ((side - w)//2, (side - h)//2))
        return preprocess_image(sq)

    def _smoothed_prediction(self, new_label, new_conf):
        self.pred_queue.append((new_label, new_conf))
        if not self.pred_queue:
            return new_label, new_conf
        
        tally = {}
        for lab, cf in self.pred_queue:
            tally[lab] = tally.get(lab, 0.0) + cf
        
        best_lab = max(tally.items(), key=lambda kv: kv[1])[0]
        total = sum(tally.values()) + 1e-8
        conf_est = tally[best_lab] / total
        return best_lab, float(conf_est)

    def on_guess(self):
        x = self._capture_vector()
        pred, conf = self.model.predict_with_conf(x)
        
        if pred is None:
            self.lbl_pred.config(text="(no data)")
            self.lbl_conf.config(text="Confidence: 0%")
            self.conf_bar.set_value(0)
            return

        self.stats["guesses"] += 1

        pred_s, conf_s = self._smoothed_prediction(pred, conf)
        self.lbl_pred.config(text=pred_s)
        self.lbl_conf.config(text=f"Confidence: {int(conf_s*100)}%")
        self.conf_bar.set_value(int(conf_s * 100))

        if pred_s == self.target and conf_s >= self.conf_threshold:
            self.score += 1
            self.stats["correct"] += 1
            self.stats["conf_sum_correct"] += conf_s
            self.stats["per_label"][self.target]["correct"] += 1
            self.lbl_score.config(text=str(self.score))
            
            # Update progress
            self.progress['correct_guesses'] += 1
            self.progress['current_streak'] += 1
            if self.progress['current_streak'] > self.progress['best_streak']:
                self.progress['best_streak'] = self.progress['current_streak']
            
            # High confidence win
            if conf_s >= 0.95:
                self.progress['high_conf_wins'] = self.progress.get('high_conf_wins', 0) + 1
            
            # Update per-category
            if self.target not in self.progress['per_category']:
                self.progress['per_category'][self.target] = {'asked': 0, 'correct': 0}
            self.progress['per_category'][self.target]['correct'] += 1
            
            self._save_progress()
            self._check_achievements()
            
            # Visual feedback
            self._show_success_feedback()
            self.root.after(800, self._advance_round)
        else:
            # Reset streak on wrong guess
            self.progress['current_streak'] = 0

    def _show_success_feedback(self):
        """Flash success animation"""
        overlay = tk.Frame(self.canvas, bg=THEME["success"])
        overlay.place(relwidth=1, relheight=1)
        overlay.lift()
        
        label = tk.Label(overlay, text="✓ Correct!", font=("Segoe UI", 24, "bold"),
                        fg="white", bg=THEME["success"])
        label.place(relx=0.5, rely=0.5, anchor="center")
        
        def fade_out(alpha=100):
            if alpha > 0:
                self.root.after(10, lambda: fade_out(alpha - 5))
            else:
                overlay.destroy()
        
        self.root.after(600, lambda: fade_out())

    def on_teach(self):
        x = self._capture_vector()
        self.X = np.vstack([self.X, x]) if self.X is not None else np.array([x])
        self.y = np.append(self.y, self.target) if self.y is not None else np.array([self.target], dtype=object)
        self.model.fit(self.X, self.y)
        
        # Update progress
        self.progress['drawings_taught'] += 1
        self._save_progress()
        self._check_achievements()
        
        self.lbl_pred.config(text=f"✓ Taught: {self.target}")
        self.lbl_conf.config(text="Added to training data")
        self.conf_bar.set_value(100)
        self.pred_queue.clear()

    def next_target(self, auto_clear=False):
        self.target = random.choice(self.active_categories)
        self.lbl_target.config(text=self.target)
        self.stats["per_label"][self.target]["asked"] += 1
        
        # Update progress per-category
        if self.target not in self.progress['per_category']:
            self.progress['per_category'][self.target] = {'asked': 0, 'correct': 0}
        self.progress['per_category'][self.target]['asked'] += 1
        
        if auto_clear:
            self.clear_canvas()

    def _advance_round(self):
        self.round_index += 1
        self.progress['rounds_won'] += 1
        self._save_progress()
        
        if self.round_index >= self.rounds_total:
            self._end_round()
        else:
            self.next_target(auto_clear=True)
            self._check_achievements()

    # -------------- End Game --------------
    def _end_round(self):
        duration = time.time() - (self.stats["started_at"] or time.time())
        total_rounds = self.rounds_total
        guesses = self.stats["guesses"]
        correct = self.stats["correct"]
        accuracy = (correct / guesses * 100.0) if guesses else 0.0
        avg_conf = (self.stats["conf_sum_correct"] / correct) if correct else 0.0

        # Update global progress
        self.progress['games_played'] += 1
        self.progress['total_guesses'] += guesses
        
        # Check for perfect game
        if accuracy == 100.0 and guesses > 0:
            self.progress['perfect_games'] = self.progress.get('perfect_games', 0) + 1
        
        # Check fastest game
        if duration < self.progress.get('fastest_game', 999):
            self.progress['fastest_game'] = duration
        
        self._save_progress()
        self._check_achievements()

        # Create modal
        win = tk.Toplevel(self.root)
        win.title("Game Complete!")
        win.configure(bg=THEME["bg"])
        win.transient(self.root)
        win.grab_set()
        win.geometry("550x700")

        container = tk.Frame(win, bg=THEME["card"], padx=32, pady=32)
        container.pack(fill="both", expand=True, padx=20, pady=20)

        # Title
        ttk.Label(container, text="🏆 Game Complete!",
                 font=("Segoe UI", 20, "bold"), foreground=THEME["success"],
                 background=THEME["card"]).pack(pady=(0, 20))

        # Achievement notifications
        if self.unlocked_this_session:
            ach_banner = tk.Frame(container, bg=THEME["success"], padx=16, pady=12)
            ach_banner.pack(fill="x", pady=(0, 16))
            
            tk.Label(ach_banner, text=f"🎉 {len(self.unlocked_this_session)} New Achievement(s) Unlocked!",
                    font=("Segoe UI", 11, "bold"), fg="white", bg=THEME["success"]).pack()

        # Stats grid
        stats_frame = tk.Frame(container, bg=THEME["card_hover"], padx=20, pady=16)
        stats_frame.pack(fill="x", pady=(0, 16))

        stats_data = [
            ("Score", f"{correct}/{total_rounds}"),
            ("Accuracy", f"{accuracy:.1f}%"),
            ("Total Guesses", str(guesses)),
            ("Avg Confidence", f"{int(avg_conf*100)}%"),
            ("Time", f"{int(duration)}s")
        ]

        for i, (label, value) in enumerate(stats_data):
            row_frame = tk.Frame(stats_frame, bg=THEME["card_hover"])
            row_frame.pack(fill="x", pady=4)
            
            tk.Label(row_frame, text=label, font=("Segoe UI", 10),
                    fg=THEME["muted"], bg=THEME["card_hover"], anchor="w").pack(side="left")
            tk.Label(row_frame, text=value, font=("Segoe UI", 12, "bold"),
                    fg=THEME["text"], bg=THEME["card_hover"], anchor="e").pack(side="right")

        # Per-label breakdown
        ttk.Label(container, text="Category Breakdown",
                 font=("Segoe UI", 12, "bold"), foreground=THEME["text"],
                 background=THEME["card"]).pack(anchor="w", pady=(8, 8))

        table_frame = tk.Frame(container, bg=THEME["card"])
        table_frame.pack(fill="both", expand=True)

        # Headers
        header_frame = tk.Frame(table_frame, bg=THEME["border"], padx=12, pady=8)
        header_frame.pack(fill="x")
        header_frame.columnconfigure(0, weight=2)
        header_frame.columnconfigure(1, weight=1)
        header_frame.columnconfigure(2, weight=1)

        for i, text in enumerate(["Category", "Asked", "Correct"]):
            tk.Label(header_frame, text=text, font=("Segoe UI", 9, "bold"),
                    fg=THEME["muted"], bg=THEME["border"]).grid(row=0, column=i, sticky="w", padx=8)

        # Data rows
        for lab in sorted(self.stats["per_label"].keys()):
            row_data = self.stats["per_label"][lab]
            row_frame = tk.Frame(table_frame, bg=THEME["card_hover"], padx=12, pady=6)
            row_frame.pack(fill="x", pady=1)
            row_frame.columnconfigure(0, weight=2)
            row_frame.columnconfigure(1, weight=1)
            row_frame.columnconfigure(2, weight=1)

            tk.Label(row_frame, text=lab, font=("Segoe UI", 10),
                    fg=THEME["text"], bg=THEME["card_hover"], anchor="w").grid(row=0, column=0, sticky="w", padx=8)
            tk.Label(row_frame, text=str(row_data["asked"]), font=("Segoe UI", 10),
                    fg=THEME["text_secondary"], bg=THEME["card_hover"]).grid(row=0, column=1, sticky="w", padx=8)
            tk.Label(row_frame, text=str(row_data["correct"]), font=("Segoe UI", 10),
                    fg=THEME["success"], bg=THEME["card_hover"]).grid(row=0, column=2, sticky="w", padx=8)

        # Buttons
        btn_frame = tk.Frame(container, bg=THEME["card"])
        btn_frame.pack(fill="x", pady=(20, 0))

        RoundedButton(btn_frame, text="🔄 Play Again", width=140, height=44,
                     style="accent", command=lambda: [win.destroy(), self._start_game()]).pack(side="left", padx=(0, 8))
        
        RoundedButton(btn_frame, text="📊 Progress", width=140, height=44,
                     command=lambda: [win.destroy(), self._show_progress()]).pack(side="left", padx=(0, 8))
        
        RoundedButton(btn_frame, text="⬅️ Menu", width=100, height=44,
                     command=lambda: [win.destroy(), self._exit_to_start()]).pack(side="left")

    # -------------- Data I/O --------------
    def save_dataset(self):
        try:
            path = filedialog.asksaveasfilename(
                defaultextension=".npz",
                initialfile=DATA_FILE,
                filetypes=[("NumPy Zip", "*.npz")])
            if not path:
                return
            np.savez_compressed(path, X=self.X, y=self.y)
            messagebox.showinfo("Success", f"Dataset saved!\n{len(self.y)} samples")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_dataset(self):
        try:
            path = filedialog.askopenfilename(filetypes=[("NumPy Zip", "*.npz")])
            if not path:
                return
            data = np.load(path, allow_pickle=True)
            self.X = data["X"]
            self.y = data["y"]
            
            mask = np.array([lab in self.active_categories for lab in self.y], dtype=bool)
            if mask.any():
                self.X = self.X[mask]
                self.y = self.y[mask]
            
            self.model.fit(self.X, self.y)
            messagebox.showinfo("Success", f"Dataset loaded!\n{len(self.y)} samples")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # -------------- Shortcuts --------------
    def _bind_shortcuts(self):
        self.root.bind("<space>", lambda e: self.on_guess())
        self.root.bind("<Key-t>", lambda e: self.on_teach())
        self.root.bind("<Key-T>", lambda e: self.on_teach())
        self.root.bind("<Key-n>", lambda e: self.next_target())
        self.root.bind("<Key-N>", lambda e: self.next_target())
        self.root.bind("<Key-c>", lambda e: self.clear_canvas())
        self.root.bind("<Key-C>", lambda e: self.clear_canvas())
        self.root.bind("<Key-b>", lambda e: self._set_mode("brush"))
        self.root.bind("<Key-B>", lambda e: self._set_mode("brush"))
        self.root.bind("<Key-e>", lambda e: self._set_mode("eraser"))
        self.root.bind("<Key-E>", lambda e: self._set_mode("eraser"))

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1920x1080")
    app = DoodleGameApp(root)
    root.mainloop()