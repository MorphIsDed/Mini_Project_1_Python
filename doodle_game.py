"""
A Doodle Recognition Game using a simple K-Nearest Neighbors classifier.
The game allows users to draw simple doodles which the AI tries to recognize.
Features:
- Real-time drawing on canvas
- Simple KNN-based recognition
- Training capability
- Save/Load dataset functionality
"""

import tkinter as tk
from tkinter import messagebox, filedialog
import random
import os
import io
import math
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFilter

# ----------------------------
# CONFIG
# ----------------------------
CANVAS_SIZE = 400
BRUSH_SIZE = 16
PREP_SIZE = 28
CATEGORIES = ["sun", "cloud", "smiley", "star", "tree", "house"]
DATA_FILE = "doodle_data.npz"
K_NEIGHBORS = 3


# ----------------------------
# Simple KNN (no sklearn)
# ----------------------------
class TinyKNN:
    """
    A simple K-Nearest Neighbors classifier implementation.
    
    Attributes:
        k (int): Number of neighbors to use for prediction
        X (np.array): Training data features of shape (N, D)
        y (np.array): Training data labels of shape (N,)
    """
    def __init__(self, k=3):
        self.k = k
        self.X = None  # shape: (N, D)
        self.y = None  # shape: (N,)

    def fit(self, X, y):
        if X is None or len(X) == 0:
            self.X = None
            self.y = None
            return
        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y, dtype=object)

    def predict_with_conf(self, x):
        """
        Predict the class of input x with confidence score.
        
        Args:
            x (np.array): Input feature vector
            
        Returns:
            tuple: (predicted_label, confidence_score)
        """
        if self.X is None or len(self.X) == 0:
            return None, 0.0
        # Euclidean distances
        dists = np.linalg.norm(self.X - x, axis=1)
        idx = np.argsort(dists)[:min(self.k, len(dists))]
        neighbors = self.y[idx]
        votes = {}
        for lab in neighbors:
            votes[lab] = votes.get(lab, 0) + 1
        # majority vote
        label = max(votes.items(), key=lambda kv: kv[1])[0]
        conf = votes[label] / len(idx)
        # distance-based softening (optional)
        # If best neighbor very far, reduce confidence a bit
        best_dist = dists[idx[0]] + 1e-6
        conf = float(conf * (1.0 / (1.0 + best_dist / 50.0)))
        return label, max(0.01, min(conf, 0.999))


# ----------------------------
# Utils: capture & preprocess
# ----------------------------
def preprocess_image(pil_img):
    """
    Preprocess a PIL image for model input.
    
    Steps:
    1. Convert to grayscale
    2. Auto-contrast for better white/black separation
    3. Resize to PREP_SIZE x PREP_SIZE
    4. Apply slight Gaussian blur
    5. Normalize pixel values to [0,1]
    
    Args:
        pil_img (PIL.Image): Input image
        
    Returns:
        np.array: Flattened preprocessed image vector
    """
    # ensure grayscale
    img = pil_img.convert("L")
    # Invert to have strokes as white on black if needed
    # Our canvas is black with white strokes already, but ensure high contrast
    img = ImageOps.autocontrast(img)
    # Resize to 28x28 using antialias
    img = img.resize((PREP_SIZE, PREP_SIZE), Image.BICUBIC)
    # Optional slight blur to smooth jaggies
    img = img.filter(ImageFilter.GaussianBlur(0.3))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # Flatten
    vec = arr.reshape(-1)
    return vec

def draw_seed_shape(label, size=PREP_SIZE):
    """
    Generate a basic prototype drawing for each category.
    Used to initialize the model with some basic examples.
    
    Args:
        label (str): Category name to generate
        size (int): Output image size
        
    Returns:
        np.array: Flattened image vector
    """
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)

    if label == "sun":
        # circle + rays
        r = size // 4
        cx, cy = size // 2, size // 2
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=200)
        for k in range(8):
            ang = 2 * math.pi * k / 8
            x1 = cx + int(r * 1.2 * math.cos(ang))
            y1 = cy + int(r * 1.2 * math.sin(ang))
            x2 = cx + int(r * 1.9 * math.cos(ang))
            y2 = cy + int(r * 1.9 * math.sin(ang))
            d.line((x1, y1, x2, y2), fill=200, width=2)

    elif label == "cloud":
        # a few overlapping circles
        centers = [(size*0.35, size*0.55), (size*0.5, size*0.45), (size*0.65, size*0.55)]
        r = size * 0.2
        for (cx, cy) in centers:
            d.ellipse((cx-r, cy-r, cx+r, cy+r), fill=200)

    elif label == "smiley":
        r = size//3
        cx, cy = size//2, size//2
        d.ellipse((cx-r, cy-r, cx+r, cy+r), outline=200, width=2)
        # eyes
        er = max(1, size//30)
        d.ellipse((cx-r//2-er, cy-r//3-er, cx-r//2+er, cy-r//3+er), fill=200)
        d.ellipse((cx+r//2-er, cy-r//3-er, cx+r//2+er, cy-r//3+er), fill=200)
        # smile
        d.arc((cx-r//2, cy-r//3, cx+r//2, cy+r//2), start=20, end=160, fill=200, width=2)

    elif label == "star":
        # simple 5-point star
        R = size*0.35
        r = size*0.15
        cx, cy = size/2, size/2
        pts = []
        for i in range(10):
            ang = -math.pi/2 + i * math.pi/5
            rad = R if i % 2 == 0 else r
            pts.append((cx + rad*math.cos(ang), cy + rad*math.sin(ang)))
        d.polygon(pts, outline=200)

    elif label == "tree":
        # trunk
        d.rectangle((size*0.45, size*0.55, size*0.55, size*0.9), fill=200)
        # canopy
        d.ellipse((size*0.25, size*0.2, size*0.75, size*0.7), outline=200, width=2)

    elif label == "house":
        # square + triangle roof
        d.rectangle((size*0.25, size*0.45, size*0.75, size*0.85), outline=200, width=2)
        d.polygon([(size*0.25, size*0.45), (size*0.5, size*0.15), (size*0.75, size*0.45)], outline=200)

    return np.asarray(img, dtype=np.float32).reshape(-1) / 255.0

def generate_seed_dataset(categories):
    X, y = [], []
    # multiple variants per label (scaled/shifted a bit)
    rng = np.random.default_rng(42)
    for lab in categories:
        base = draw_seed_shape(lab, PREP_SIZE).reshape(PREP_SIZE, PREP_SIZE)
        for _ in range(8):  # small seed count
            img = Image.fromarray((base*255).astype(np.uint8), mode="L")
            # random affine-ish jitter by paste on slightly shifted canvas
            shift_x = rng.integers(-2, 3)
            shift_y = rng.integers(-2, 3)
            canvas = Image.new("L", (PREP_SIZE+4, PREP_SIZE+4), 0)
            canvas.paste(img, (2+shift_x, 2+shift_y))
            canvas = canvas.resize((PREP_SIZE, PREP_SIZE), Image.BICUBIC)
            arr = np.asarray(canvas, dtype=np.float32) / 255.0
            X.append(arr.reshape(-1))
            y.append(lab)
    return np.array(X, dtype=np.float32), np.array(y, dtype=object)

# ----------------------------
# Main App
# ----------------------------
class DoodleGameApp:
    """
    Main game application class using tkinter.
    
    Features:
    - Drawing canvas
    - Real-time prediction
    - Training capability
    - Score tracking
    - Dataset save/load
    
    Key bindings:
    - Space: Make prediction
    - T: Train current drawing
    - N: New target
    - C: Clear canvas
    """
    
    def __init__(self, root):
        """
        Initialize the game window and components.
        
        Args:
            root (tk.Tk): Root window
        """
        self.root = root
        self.root.title("Doodle Guess Game - AI + Image Processing")

        self.model = TinyKNN(k=K_NEIGHBORS)
        self.X, self.y = generate_seed_dataset(CATEGORIES)
        self.model.fit(self.X, self.y)

        self.score = 0
        self.target = random.choice(CATEGORIES)

        # Canvas and PIL mirror for clean capture
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black", cursor="tcross")
        self.canvas.grid(row=0, column=0, columnspan=6, padx=10, pady=10)

        self._last = None
        self.canvas.bind("<ButtonPress-1>", self.on_draw_start)
        self.canvas.bind("<B1-Motion>", self.on_draw_move)

        # We draw to an offscreen PIL image for crisp preprocessing
        self.pil_canvas = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_canvas)

        # Controls
        self.lbl_target = tk.Label(root, text=f"Target: {self.target}", font=("Segoe UI", 14, "bold"))
        self.lbl_target.grid(row=1, column=0, sticky="w", padx=10)

        self.lbl_pred = tk.Label(root, text=f"Prediction: -", font=("Segoe UI", 12))
        self.lbl_pred.grid(row=1, column=1, sticky="w", padx=10)

        self.lbl_conf = tk.Label(root, text=f"Confidence: -", font=("Segoe UI", 12))
        self.lbl_conf.grid(row=1, column=2, sticky="w", padx=10)

        self.lbl_score = tk.Label(root, text=f"Score: {self.score}", font=("Segoe UI", 12, "bold"))
        self.lbl_score.grid(row=1, column=3, sticky="w", padx=10)

        btn_guess = tk.Button(root, text="Guess (Space)", command=self.on_guess, width=15)
        btn_guess.grid(row=2, column=0, pady=6)

        btn_teach = tk.Button(root, text="Teach as Target (T)", command=self.on_teach, width=18)
        btn_teach.grid(row=2, column=1, pady=6)

        btn_next = tk.Button(root, text="New Target (N)", command=self.next_target, width=15)
        btn_next.grid(row=2, column=2, pady=6)

        btn_clear = tk.Button(root, text="Clear (C)", command=self.clear_canvas, width=12)
        btn_clear.grid(row=2, column=3, pady=6)

        btn_save = tk.Button(root, text="Save Dataset", command=self.save_dataset, width=12)
        btn_save.grid(row=2, column=4, pady=6)

        btn_load = tk.Button(root, text="Load Dataset", command=self.load_dataset, width=12)
        btn_load.grid(row=2, column=5, pady=6)

        # Shortcuts
        root.bind("<space>", lambda e: self.on_guess())
        root.bind("<Key-t>", lambda e: self.on_teach())
        root.bind("<Key-T>", lambda e: self.on_teach())
        root.bind("<Key-c>", lambda e: self.clear_canvas())
        root.bind("<Key-C>", lambda e: self.clear_canvas())
        root.bind("<Key-n>", lambda e: self.next_target())
        root.bind("<Key-N>", lambda e: self.next_target())

        # Hint footer
        self.hint = tk.Label(root, text="Draw with mouse. Space=Guess | T=Teach | N=New target | C=Clear", fg="#666")
        self.hint.grid(row=3, column=0, columnspan=6, pady=(0, 10))

    # ---- Drawing ----
    def on_draw_start(self, event):
        self._last = (event.x, event.y)
        self.draw_point(event.x, event.y)

    def on_draw_move(self, event):
        if self._last is not None:
            x0, y0 = self._last
            x1, y1 = event.x, event.y
            self.canvas.create_line(x0, y0, x1, y1, fill="white", width=BRUSH_SIZE, capstyle=tk.ROUND, smooth=True)
            # Draw on PIL image, thicker line
            self.pil_draw.line((x0, y0, x1, y1), fill=255, width=BRUSH_SIZE)
            self._last = (x1, y1)

    def draw_point(self, x, y):
        r = BRUSH_SIZE // 2
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.pil_draw.ellipse((x - r, y - r, x + r, y + r), fill=255)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.canvas.configure(bg="black")
        self.pil_canvas = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_canvas)
        self.lbl_pred.config(text="Prediction: -")
        self.lbl_conf.config(text="Confidence: -")

    # ---- Game / Model ----
    def capture_vector(self):
        """
        Process the current canvas drawing for prediction.
        
        Steps:
        1. Capture canvas content
        2. Find drawing bounding box
        3. Crop and center the drawing
        4. Preprocess for model input
        
        Returns:
            np.array: Processed feature vector
        """
        # Trim & center the drawing to be robust
        img = self.pil_canvas.copy()
        # Find bounding box of strokes
        arr = np.array(img)
        ys, xs = np.where(arr > 10)
        if len(xs) > 0 and len(ys) > 0:
            minx, maxx = xs.min(), xs.max()
            miny, maxy = ys.min(), ys.max()
            # add margin
            minx = max(0, minx - 10); miny = max(0, miny - 10)
            maxx = min(arr.shape[1]-1, maxx + 10); maxy = min(arr.shape[0]-1, maxy + 10)
            img = img.crop((minx, miny, maxx, maxy))
        # Pad to square
        w, h = img.size
        side = max(w, h)
        sq = Image.new("L", (side, side), 0)
        sq.paste(img, ((side - w)//2, (side - h)//2))
        return preprocess_image(sq)

    def on_guess(self):
        """
        Handle prediction button/hotkey.
        - Captures current drawing
        - Makes prediction
        - Updates score if correct
        - Shows feedback message
        """
        x = self.capture_vector()
        pred, conf = self.model.predict_with_conf(x)
        if pred is None:
            self.lbl_pred.config(text="Prediction: (no data)")
            self.lbl_conf.config(text="Confidence: 0.00")
            return
        self.lbl_pred.config(text=f"Prediction: {pred}")
        self.lbl_conf.config(text=f"Confidence: {conf:.2f}")

        if pred == self.target and conf >= 0.25:
            self.score += 1
            self.lbl_score.config(text=f"Score: {self.score}")
            messagebox.showinfo("Nice!", f"Correct! It guessed '{pred}'. +1 point 🎉")
            self.next_target(auto_clear=True)

    def on_teach(self):
        """
        Handle teach button/hotkey.
        Adds current drawing to training set with current target label.
        """
        x = self.capture_vector()
        self.X = np.vstack([self.X, x]) if self.X is not None else np.array([x])
        self.y = np.append(self.y, self.target) if self.y is not None else np.array([self.target], dtype=object)
        self.model.fit(self.X, self.y)
        self.lbl_pred.config(text=f"Taught: {self.target}")
        self.lbl_conf.config(text=f"Confidence: -")

    def next_target(self, auto_clear=False):
        """
        Select new random target category.
        
        Args:
            auto_clear (bool): Whether to clear canvas automatically
        """
        self.target = random.choice(CATEGORIES)
        self.lbl_target.config(text=f"Target: {self.target}")
        if auto_clear:
            self.clear_canvas()

    # ---- Persistence ----
    def save_dataset(self):
        """Save current training data to NPZ file."""
        try:
            path = filedialog.asksaveasfilename(defaultextension=".npz", initialfile=DATA_FILE,
                                                filetypes=[("NumPy Zip", "*.npz")])
            if not path:
                return
            np.savez_compressed(path, X=self.X, y=self.y)
            messagebox.showinfo("Saved", f"Dataset saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_dataset(self):
        """Load training data from NPZ file."""
        try:
            path = filedialog.askopenfilename(filetypes=[("NumPy Zip", "*.npz")])
            if not path:
                return
            data = np.load(path, allow_pickle=True)
            self.X = data["X"]
            self.y = data["y"]
            self.model.fit(self.X, self.y)
            messagebox.showinfo("Loaded", f"Dataset loaded from:\n{path}\nSamples: {len(self.y)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))



# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = DoodleGameApp(root)
    root.mainloop()