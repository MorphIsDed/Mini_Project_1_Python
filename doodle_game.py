"""
Doodle Recognition Game
======================

A machine learning-based drawing recognition game where users draw simple doodles 
that an AI model tries to recognize in real-time.

Key Features:
    - Real-time drawing recognition using KNN classifier
    - Multiple difficulty levels
    - Drawing tools (brush, eraser, colors)
    - Training capability (teach mode)
    - Score tracking and statistics
    - Dataset save/load functionality

Game Modes:
    - Easy: 4 categories, lower confidence threshold
    - Medium: 6 categories, moderate threshold
    - Hard: 6 categories, high threshold

Controls:
    - Space: Make prediction
    - T: Teach current drawing
    - N: New target
    - C: Clear canvas
    - B: Switch to brush
    - E: Switch to eraser
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
CANVAS_SIZE = 520      # Size of drawing canvas in pixels
PREP_SIZE = 28         # Size of preprocessed images for model
DATA_FILE = "doodle_data.npz"
K_NEIGHBORS = 5        # Number of neighbors for KNN

# Available drawing categories
ALL_CATEGORIES = ["sun", "cloud", "smiley", "star", "tree", "house"]

# Difficulty presets with their parameters
DIFF_PRESETS = {
    "Easy":   {"categories": 4, "conf_threshold": 0.25, "smooth_window": 5},
    "Medium": {"categories": 6, "conf_threshold": 0.38, "smooth_window": 7},
    "Hard":   {"categories": 6, "conf_threshold": 0.55, "smooth_window": 9},
}

# UI Theme colors
THEME = {
    "bg": "#0f1117",       # Background
    "card": "#171a21",     # Card/panel background
    "accent": "#5b8cff",   # Primary accent color
    "text": "#e5e7eb",     # Primary text
    "muted": "#9aa3af",    # Secondary text
    "border": "#2a2f39",   # Border color
}

# Available color palette
PALETTE = [
    ("White", "#ffffff"),
    ("Yellow", "#fbbf24"),
    ("Red", "#ef4444"),
    ("Green", "#22c55e"),
    ("Blue", "#3b82f6"),
    ("Cyan", "#06b6d4"),
    ("Magenta", "#e879f9"),
]

# ----------------------------
# KNN Classifier Implementation
# ----------------------------
class TinyKNN:
    """
    A lightweight K-Nearest Neighbors classifier with distance weighting.
    
    Features:
        - Euclidean + Cosine similarity hybrid distance metric
        - Distance-weighted voting
        - Confidence estimation
        - Class frequency normalization
    """
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
        w = 1.0 / (d[idx] + 1e-6)  # distance weights
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
# Image Processing Utilities
# ----------------------------
def preprocess_image(pil_img):
    """
    Preprocess a drawing for model input.
    
    Steps:
    1. Convert to grayscale
    2. Auto-contrast for better white/black separation
    3. Resize to PREP_SIZE x PREP_SIZE
    4. Apply slight Gaussian blur for smoothing
    5. Normalize pixel values to [0,1]
    
    Args:
        pil_img (PIL.Image): Input image
        
    Returns:
        np.array: Flattened preprocessed image vector
    """
    img = pil_img.convert("L")
    img = ImageOps.autocontrast(img)
    img = img.resize((PREP_SIZE, PREP_SIZE), Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(0.35))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.reshape(-1)

def draw_seed_shape(label, size=PREP_SIZE):
    """
    Generate prototype drawings for initial training data.
    Each shape is procedurally generated based on the category.
    
    Args:
        label (str): Category name
        size (int): Output image size
        
    Returns:
        np.array: Flattened image vector [0,1] range
    """
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
    """
    Create initial training dataset with synthetic examples.
    Applies random transformations to add variety.
    
    Args:
        categories (list): List of category names
        
    Returns:
        tuple: (X, y) - feature vectors and labels
    """
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
# Main Application
# ----------------------------
class DoodleGameApp:
    """
    Main game application class implementing the GUI and game logic.
    
    Features:
    - Drawing canvas with tools
    - Real-time prediction
    - Score tracking
    - Multiple rounds
    - Statistics
    - Dataset management
    """
    
    def __init__(self, root):
        """
        Initialize game window and components.
        
        Args:
            root (tk.Tk): Root window instance
        """
        self.root = root
        self.root.title("Doodle Guess")
        self._apply_theme()

        # ---- Global state ----
        self.brush_size = 16
        self.brush_color = "#ffffff"
        self.mode = tk.StringVar(value="brush")  # brush | eraser
        self._last = None

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

        # ---- Layout ----
        self.container = ttk.Frame(self.root, padding=0)
        self.container.pack(fill="both", expand=True)
        self.start_frame = self._build_start_frame(self.container)
        self.game_frame = self._build_game_frame(self.container)
        self._show_frame(self.start_frame)

    # -------------- THEME --------------
    def _apply_theme(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure(".", background=THEME["bg"], foreground=THEME["text"], fieldbackground=THEME["card"])
        s.configure("Card.TFrame", background=THEME["card"])
        s.configure("Title.TLabel", font=("Segoe UI", 20, "bold"), background=THEME["bg"], foreground=THEME["text"])
        s.configure("H2.TLabel", font=("Segoe UI", 13, "bold"), background=THEME["card"])
        s.configure("Muted.TLabel", font=("Segoe UI", 10), foreground=THEME["muted"], background=THEME["card"])
        s.configure("MutedGlobal.TLabel", font=("Segoe UI", 10), foreground=THEME["muted"], background=THEME["bg"])
        s.configure("TButton", padding=8, relief="flat", background=THEME["card"], foreground=THEME["text"])
        s.map("TButton", background=[("active", "#202531")])
        s.configure("Accent.TButton", background=THEME["accent"], foreground="white")
        s.map("Accent.TButton", background=[("active", "#486fdd")])
        s.configure("Danger.TButton", background="#ef4444", foreground="white")
        s.map("Danger.TButton", background=[("active", "#c93131")])
        s.configure("TScale", troughcolor="#0b0e14")
        s.configure("Horizontal.TProgressbar", troughcolor="#0b0e14", background=THEME["accent"])

    # -------------- SCREENS --------------
    def _show_frame(self, frame):
        for child in self.container.winfo_children():
            child.pack_forget()
        frame.pack(fill="both", expand=True)

    def _build_start_frame(self, parent):
        f = ttk.Frame(parent, padding=24)
        f.columnconfigure(0, weight=1)

        ttk.Label(f, text="Doodle Guess", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(f, text="AI-powered doodle classification game", style="MutedGlobal.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 16))

        card = ttk.Frame(f, style="Card.TFrame", padding=18)
        card.grid(row=2, column=0, sticky="nsew")
        for c in range(4): card.columnconfigure(c, weight=1)

        # Difficulty
        ttk.Label(card, text="Difficulty", style="H2.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.diff_var = tk.StringVar(value="Medium")
        for i, name in enumerate(["Easy", "Medium", "Hard"]):
            rb = ttk.Radiobutton(card, text=name, value=name, variable=self.diff_var)
            rb.grid(row=1, column=i, sticky="w", padx=(0, 12))

        # Rounds
        ttk.Label(card, text="Rounds", style="H2.TLabel").grid(row=2, column=0, sticky="w", pady=(16, 6))
        self.rounds_var = tk.IntVar(value=10)
        try:
            # ttk.Spinbox in Py3.8+
            rounds_input = ttk.Spinbox(card, from_=3, to=50, textvariable=self.rounds_var, width=6)
        except Exception:
            rounds_input = tk.Spinbox(card, from_=3, to=50, textvariable=self.rounds_var, width=6)
        rounds_input.grid(row=3, column=0, sticky="w")

        ttk.Label(card, text="Easy uses fewer classes and a looser confidence requirement. Hard uses all classes and stricter confidence.", style="Muted.TLabel").grid(row=4, column=0, columnspan=4, sticky="w", pady=(14,0))

        start_btn = ttk.Button(f, text="Start", style="Accent.TButton", command=self._start_game)
        start_btn.grid(row=3, column=0, sticky="w", pady=(16, 0))

        return f

    def _build_game_frame(self, parent):
        f = ttk.Frame(parent, padding=12)
        # Header
        header = ttk.Frame(f, style="Card.TFrame", padding=14)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Doodle Guess", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        self.lbl_score = ttk.Label(header, text="Score: 0", style="H2.TLabel")
        self.lbl_score.grid(row=0, column=1, sticky="e")

        # Left: canvas + toolbar
        left = ttk.Frame(f, padding=(0, 12))
        left.grid(row=1, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)
        f.columnconfigure(0, weight=1)
        f.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(left, style="Card.TFrame", padding=10)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        toolbar.columnconfigure(5, weight=1)

        self.btn_brush = ttk.Button(toolbar, text="Brush", command=lambda: self._set_mode("brush"))
        self.btn_brush.grid(row=0, column=0, padx=(0, 6))
        self.btn_eraser = ttk.Button(toolbar, text="Eraser", command=lambda: self._set_mode("eraser"))
        self.btn_eraser.grid(row=0, column=1, padx=(0, 12))

        ttk.Label(toolbar, text="Size", style="Muted.TLabel").grid(row=0, column=2, padx=(0, 6))
        self.size_var = tk.IntVar(value=self.brush_size)
        size_scale = ttk.Scale(toolbar, from_=4, to=64, orient="horizontal",
                               command=self._on_size_change, variable=self.size_var)
        size_scale.grid(row=0, column=3, sticky="ew", padx=(0, 12))

        color_frame = ttk.Frame(toolbar, style="Card.TFrame")
        color_frame.grid(row=0, column=4, padx=(0, 12))
        for i, (name, hexv) in enumerate(PALETTE):
            b = tk.Button(color_frame, width=2, height=1, bg=hexv, relief="flat",
                          command=lambda hv=hexv: self._set_color(hv))
            b.grid(row=0, column=i, padx=2)
        ttk.Button(color_frame, text="Pick", command=self._pick_color).grid(row=0, column=len(PALETTE), padx=(6,0))

        ttk.Button(toolbar, text="Clear", command=self.clear_canvas, style="TButton").grid(row=0, column=5, sticky="e")
        ttk.Button(toolbar, text="Guess", command=self.on_guess, style="Accent.TButton").grid(row=0, column=6, padx=(8,0))

        # Canvas card
        canvas_card = ttk.Frame(left, style="Card.TFrame", padding=8)
        canvas_card.grid(row=1, column=0, sticky="nsew")
        canvas_card.columnconfigure(0, weight=1)
        canvas_card.rowconfigure(0, weight=1)

        border = tk.Frame(canvas_card, bg=THEME["bg"], bd=0, highlightthickness=1, highlightbackground=THEME["border"])
        border.grid(row=0, column=0, sticky="nsew")
        self.canvas = tk.Canvas(border, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="black", bd=0, highlightthickness=0, cursor="tcross")
        self.canvas.pack(padx=8, pady=8)

        # Offscreen PIL
        self.pil_canvas = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_canvas)

        self.canvas.bind("<ButtonPress-1>", self._on_draw_start)
        self.canvas.bind("<B1-Motion>", self._on_draw_move)
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, "_last", None))

        # Right: sidebar
        side = ttk.Frame(f, style="Card.TFrame", padding=14)
        side.grid(row=1, column=1, sticky="ns")
        for r in range(10): side.rowconfigure(r, weight=0)

        ttk.Label(side, text="Target", style="H2.TLabel").grid(row=0, column=0, sticky="w")
        self.lbl_target = ttk.Label(side, text="—", font=("Segoe UI", 13, "bold"))
        self.lbl_target.grid(row=1, column=0, sticky="w", pady=(2, 10))

        ttk.Label(side, text="Prediction", style="H2.TLabel").grid(row=2, column=0, sticky="w")
        self.lbl_pred = ttk.Label(side, text="—", font=("Segoe UI", 12))
        self.lbl_pred.grid(row=3, column=0, sticky="w", pady=(2, 4))

        self.conf_bar = ttk.Progressbar(side, orient="horizontal", mode="determinate", length=220, maximum=100)
        self.conf_bar.grid(row=4, column=0, sticky="ew", pady=(0, 2))
        self.lbl_conf = ttk.Label(side, text="Confidence: —", style="Muted.TLabel")
        self.lbl_conf.grid(row=5, column=0, sticky="w", pady=(0, 10))

        ttk.Button(side, text="Teach as Target", command=self.on_teach).grid(row=6, column=0, sticky="ew", pady=(2, 6))
        ttk.Button(side, text="New Target", command=self.next_target).grid(row=7, column=0, sticky="ew", pady=(0, 10))

        ttk.Button(side, text="Save Dataset", command=self.save_dataset).grid(row=8, column=0, sticky="ew", pady=(0, 4))
        ttk.Button(side, text="Load Dataset", command=self.load_dataset).grid(row=9, column=0, sticky="ew", pady=(0, 12))

        ttk.Button(side, text="Exit to Start", command=self._exit_to_start).grid(row=10, column=0, sticky="ew")

        # Status
        status = ttk.Frame(f, style="Card.TFrame", padding=(12, 8))
        status.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(
            status,
            text="Shortcuts: Space=Guess • T=Teach • N=New Target • C=Clear • B=Brush • E=Eraser",
            style="Muted.TLabel"
        ).grid(row=0, column=0, sticky="w")

        self._bind_shortcuts()
        return f

    # -------------- START / RESET --------------
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

        # model
        self.model = TinyKNN(k=K_NEIGHBORS)
        X, y = generate_seed_dataset(self.active_categories)
        self.X, self.y = X, y
        self.model.fit(self.X, self.y)

        # reset game
        self.score = 0
        self.round_index = 0
        self.lbl_score.config(text="Score: 0")
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
        self._show_frame(self.start_frame)

    # -------------- DRAWING --------------
    def _on_draw_start(self, event):
        """Handle mouse button press - start drawing."""
        self._last = (event.x, event.y)
        self._draw_point(event.x, event.y)

    def _on_draw_move(self, event):
        """Handle mouse drag - continue drawing."""
        if self._last is None: return
        x0, y0 = self._last; x1, y1 = event.x, event.y
        if self.mode.get() == "eraser":
            color_canvas = "black"; color_pil = 0
        else:
            color_canvas = self.brush_color; color_pil = 255
        self.canvas.create_line(x0, y0, x1, y1, fill=color_canvas,
                                width=self.brush_size, capstyle=tk.ROUND, smooth=True)
        self.pil_draw.line((x0, y0, x1, y1), fill=color_pil, width=self.brush_size)
        self._last = (x1, y1)

    def _draw_point(self, x, y):
        r = self.brush_size // 2
        if self.mode.get() == "eraser":
            color_canvas = "black"; color_pil = 0
        else:
            color_canvas = self.brush_color; color_pil = 255
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=color_canvas, outline=color_canvas)
        self.pil_draw.ellipse((x - r, y - r, x + r, y + r), fill=color_pil)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.canvas.configure(bg="black")
        self.pil_canvas = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_canvas)
        self.lbl_pred.config(text="—")
        self.lbl_conf.config(text="Confidence: —")
        self.conf_bar["value"] = 0
        self.pred_queue.clear()

    def _on_size_change(self, val):
        try:
            self.brush_size = int(float(val))
        except Exception:
            pass

    def _set_color(self, hexv):
        self.brush_color = hexv
        self._set_mode("brush")

    def _pick_color(self):
        color = colorchooser.askcolor(color=self.brush_color)[1]
        if color:
            self._set_color(color)

    def _set_mode(self, mode):
        self.mode.set(mode)
        if mode == "brush":
            self.btn_brush.configure(style="Accent.TButton")
            self.btn_eraser.configure(style="TButton")
        else:
            self.btn_eraser.configure(style="Accent.TButton")
            self.btn_brush.configure(style="TButton")

    # -------------- MODEL / GAME --------------
    def _capture_vector(self):
        """
        Process current drawing for prediction.
        Crops to content, centers, and preprocesses.
        """
        img = self.pil_canvas.copy()
        arr = np.array(img)
        ys, xs = np.where(arr > 10)
        if len(xs) > 0 and len(ys) > 0:
            minx, maxx = xs.min(), xs.max()
            miny, maxy = ys.min(), ys.max()
            minx = max(0, minx - 12); miny = max(0, miny - 12)
            maxx = min(arr.shape[1]-1, maxx + 12); maxy = min(arr.shape[0]-1, maxy + 12)
            img = img.crop((minx, miny, maxx, maxy))
        w, h = img.size
        side = max(w, h)
        sq = Image.new("L", (side, side), 0)
        sq.paste(img, ((side - w)//2, (side - h)//2))
        return preprocess_image(sq)

    def _smoothed_prediction(self, new_label, new_conf):
        """
        Apply temporal smoothing to predictions.
        Reduces jitter in real-time predictions.
        """
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
        """
        Handle prediction request.
        Updates UI, tracks stats, advances round if correct.
        """
        x = self._capture_vector()
        pred, conf = self.model.predict_with_conf(x)
        if pred is None:
            self.lbl_pred.config(text="(no data)")
            self.lbl_conf.config(text="Confidence: 0%")
            self.conf_bar["value"] = 0
            return

        # stats: a guess occurred
        self.stats["guesses"] += 1

        pred_s, conf_s = self._smoothed_prediction(pred, conf)
        self.lbl_pred.config(text=pred_s)
        self.lbl_conf.config(text=f"Confidence: {int(conf_s*100)}%")
        self.conf_bar["value"] = int(conf_s * 100)

        if pred_s == self.target and conf_s >= self.conf_threshold:
            self.score += 1
            self.stats["correct"] += 1
            self.stats["conf_sum_correct"] += conf_s
            self.lbl_score.config(text=f"Score: {self.score}")
            messagebox.showinfo("Correct", f"Recognized: {pred_s}")
            self._advance_round()

    def on_teach(self):
        x = self._capture_vector()
        self.X = np.vstack([self.X, x]) if self.X is not None else np.array([x])
        self.y = np.append(self.y, self.target) if self.y is not None else np.array([self.target], dtype=object)
        self.model.fit(self.X, self.y)
        self.lbl_pred.config(text=f"Taught: {self.target}")
        self.lbl_conf.config(text="Confidence: —")
        self.conf_bar["value"] = 0
        self.pred_queue.clear()

    def next_target(self, auto_clear=False):
        self.target = random.choice(self.active_categories)
        self.lbl_target.config(text=self.target)
        if auto_clear: self.clear_canvas()
        # track that this label was asked in this round
        self.stats["per_label"][self.target]["asked"] += 1

    def _advance_round(self):
        # One target successfully completed -> next round
        self.round_index += 1
        if self.round_index >= self.rounds_total:
            self._end_round()
        else:
            self.next_target(auto_clear=True)

    # -------------- SCOREBOARD --------------
    def _end_round(self):
        duration = time.time() - (self.stats["started_at"] or time.time())
        total_rounds = self.rounds_total
        guesses = self.stats["guesses"]
        correct = self.stats["correct"]
        accuracy = (correct / guesses * 100.0) if guesses else 0.0
        avg_conf = (self.stats["conf_sum_correct"] / correct) if correct else 0.0

        # Layout modal
        win = tk.Toplevel(self.root)
        win.title("Round Summary")
        win.configure(bg=THEME["bg"])
        win.transient(self.root)
        win.grab_set()

        container = ttk.Frame(win, padding=18, style="Card.TFrame")
        container.grid(row=0, column=0, sticky="nsew")
        win.columnconfigure(0, weight=1)
        win.rowconfigure(0, weight=1)

        ttk.Label(container, text="Round Summary", style="H2.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 12))

        # Top metrics
        top = ttk.Frame(container, style="Card.TFrame")
        top.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        for i in range(4): top.columnconfigure(i, weight=1)

        ttk.Label(top, text=f"Rounds: {total_rounds}", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(top, text=f"Correct: {correct}", style="Muted.TLabel").grid(row=0, column=1, sticky="w")
        ttk.Label(top, text=f"Guesses: {guesses}", style="Muted.TLabel").grid(row=0, column=2, sticky="w")
        ttk.Label(top, text=f"Accuracy: {accuracy:.1f}%", style="Muted.TLabel").grid(row=0, column=3, sticky="w")
        ttk.Label(top, text=f"Avg confidence (correct): {int(avg_conf*100)}%", style="Muted.TLabel").grid(row=1, column=0, columnspan=4, sticky="w", pady=(6,0))
        ttk.Label(top, text=f"Time: {int(duration)} s", style="Muted.TLabel").grid(row=2, column=0, columnspan=4, sticky="w", pady=(2,0))

        # Per-label table
        table = ttk.Frame(container, style="Card.TFrame")
        table.grid(row=2, column=0, sticky="ew")
        for i in range(3): table.columnconfigure(i, weight=1)

        header_style = "Muted.TLabel"
        ttk.Label(table, text="Label", style=header_style).grid(row=0, column=0, sticky="w", padx=(0,8))
        ttk.Label(table, text="Asked", style=header_style).grid(row=0, column=1, sticky="w", padx=(0,8))
        ttk.Label(table, text="Correct", style=header_style).grid(row=0, column=2, sticky="w")

        r = 1
        for lab in sorted(self.stats["per_label"].keys()):
            row = self.stats["per_label"][lab]
            ttk.Label(table, text=lab).grid(row=r, column=0, sticky="w", padx=(0,8))
            ttk.Label(table, text=str(row["asked"])).grid(row=r, column=1, sticky="w", padx=(0,8))
            ttk.Label(table, text=str(row["correct"])).grid(row=r, column=2, sticky="w")
            r += 1

        # Actions
        actions = ttk.Frame(container, style="Card.TFrame")
        actions.grid(row=3, column=0, sticky="ew", pady=(12,0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=0)
        actions.columnconfigure(2, weight=0)

        def restart_same():
            win.destroy()
            self._start_game()

        def back_to_start():
            win.destroy()
            self._exit_to_start()

        ttk.Button(actions, text="Play Again", style="Accent.TButton", command=restart_same).grid(row=0, column=1, padx=(0,8))
        ttk.Button(actions, text="Close", command=back_to_start).grid(row=0, column=2)

    # -------------- DATA I/O --------------
    def save_dataset(self):
        """Save current training data to NPZ file."""
        try:
            path = filedialog.asksaveasfilename(defaultextension=".npz", initialfile=DATA_FILE,
                                                filetypes=[("NumPy Zip", "*.npz")])
            if not path: return
            np.savez_compressed(path, X=self.X, y=self.y)
            messagebox.showinfo("Saved", f"Dataset saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_dataset(self):
        """Load training data from NPZ file."""
        try:
            path = filedialog.askopenfilename(filetypes=[("NumPy Zip", "*.npz")])
            if not path: return
            data = np.load(path, allow_pickle=True)
            self.X = data["X"]; self.y = data["y"]
            mask = np.array([lab in self.active_categories for lab in self.y], dtype=bool)
            if mask.any():
                self.X = self.X[mask]; self.y = self.y[mask]
            self.model.fit(self.X, self.y)
            messagebox.showinfo("Loaded", f"Dataset loaded: {len(self.y)} samples")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # -------------- SHORTCUTS --------------
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
    app = DoodleGameApp(root)
    root.mainloop()