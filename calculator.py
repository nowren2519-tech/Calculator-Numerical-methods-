import tkinter as tk
from tkinter import font as tkfont
import math
import os
from tkinter.font import BOLD


def safe_eval(expr: str, x: float) -> float:
    allowed = {
        'x': x,
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
        'exp': math.exp, 'log': math.log, 'log10': math.log10,
        'sqrt': math.sqrt, 'abs': abs,
        'pi': math.pi, 'e': math.e,
    }
    expr = expr.replace('^', '**')
    try:
        return float(eval(expr, {"__builtins__": {}}, allowed))
    except Exception as ex:
        raise ValueError(f"Bad expression: {ex}")

def numerical_derivative(expr, x):
    h = 1e-7
    return (safe_eval(expr, x + h) - safe_eval(expr, x - h)) / (2 * h)

def _rel_error(abs_err, new_val):

    if new_val == 0:
        return 0.0
    return abs(abs_err / new_val)

def _pct_str(rel_err):
    return f"{round(rel_err * 100, 5)}%"

def bisection(expr, a, b, tol, max_iter):
    fa = safe_eval(expr, a)
    fb = safe_eval(expr, b)
    if fa * fb > 0:
        return None, [], "f(a) and f(b) must have opposite signs!"
    rows = []
    prev_c = None
    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = safe_eval(expr, c)
        err = abs(b - a) / 2
        if prev_c is None:
            abs_err = abs(b - a)
        else:
            abs_err = abs(c - prev_c)
        rel_err = _rel_error(abs_err, c)
        rows.append((i, round(a,7), round(b,7), round(c,7), round(fc,7), round(err,7),
                     round(abs_err,7), round(rel_err,7), _pct_str(rel_err)))
        prev_c = c
        if err < tol or abs(fc) < 1e-14:
            break
        if fa * fc < 0:
            b = c;
            fb = fc
        else:
            a = c;
            fa = fc
    return c, rows, None

def false_position(expr, a, b, tol, max_iter):
    fa = safe_eval(expr, a)
    fb = safe_eval(expr, b)
    if fa * fb > 0:
        return None, [], "f(a) and f(b) must have opposite signs!"
    rows = []
    c = a
    prev_c = None
    for i in range(1, max_iter + 1):
        fa = safe_eval(expr, a); fb = safe_eval(expr, b)

        c = (a * fb - b * fa) / (fb - fa)
        fc = safe_eval(expr, c)
        err = abs(fc)
        if prev_c is None:
            abs_err = abs(c - a)
        else:
            abs_err = abs(c - prev_c)
        rel_err = _rel_error(abs_err, c)
        rows.append((i, round(a,7), round(b,7), round(c,7), round(fc,7), round(err,7),
                     round(abs_err,7), round(rel_err,7), _pct_str(rel_err)))
        prev_c = c
        if err < tol:
            break
        if fa * fc < 0:
            b = c
        else:
            a = c
    return c, rows, None

def newton_raphson(expr, x0, tol, max_iter):
    rows = []
    x = x0
    for i in range(1, max_iter + 1):
        fx = safe_eval(expr, x)
        fpx = numerical_derivative(expr, x)
        if abs(fpx) < 1e-14:
            return None, rows, "Derivative near zero. Choose different x0."
        x1 = x - fx / fpx
        err = abs(x1 - x)
        rel_err = _rel_error(err, x1)
        rows.append((i, round(x,7), round(fx,7), round(fpx,7), round(x1,7),
                     round(err,7), round(rel_err,7), _pct_str(rel_err)))
        x = x1
        if err < tol:
            break
    return x, rows, None

def secant(expr, x0, x1, tol, max_iter):
    rows = []
    xa, xb = x0, x1
    for i in range(1, max_iter + 1):
        fa = safe_eval(expr, xa); fb = safe_eval(expr, xb)
        if abs(fb - fa) < 1e-14:
            return None, rows, "Division by zero."
        xc = xb - fb * (xb - xa) / (fb - fa)
        err = abs(xc - xb)
        rel_err = _rel_error(err, xc)
        rows.append((i, round(xa,7), round(xb,7), round(xc,7), round(safe_eval(expr,xc),7),
                     round(err,7), round(rel_err,7), _pct_str(rel_err)))
        xa, xb = xb, xc
        if err < tol:
            break
    return xb, rows, None

def fixed_point(expr, gexpr, x0, tol, max_iter):
    rows = []
    x = x0
    for i in range(1, max_iter + 1):
        x1 = safe_eval(gexpr, x)
        err = abs(x1 - x)
        rel_err = _rel_error(err, x1)
        rows.append((i, round(x,7), round(x1,7), round(safe_eval(expr,x1),7),
                     round(err,7), round(rel_err,7), _pct_str(rel_err)))
        x = x1
        if err < tol or not math.isfinite(x):
            break
    return x, rows, None

def gaussian_elim(A, b, pivot=False):
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]
    steps = []
    for col in range(n):
        if pivot:
            mr = col
            for r in range(col+1, n):
                if abs(M[r][col]) > abs(M[mr][col]):
                    mr = r
            if mr != col:
                M[col], M[mr] = M[mr], M[col]
                steps.append(f"Swap R{col+1} ↔ R{mr+1}")
        if abs(M[col][col]) < 1e-12:
            return None, "Singular matrix."
        for row in range(col+1, n):
            f = M[row][col] / M[col][col]
            for c in range(col, n+1):
                M[row][c] -= f * M[col][c]
            steps.append(f"R{row+1} ← R{row+1} - ({f:.4f})×R{col+1}")
    x = [0.0]*n
    for i in range(n-1, -1, -1):
        x[i] = M[i][n]
        for j in range(i+1, n):
            x[i] -= M[i][j]*x[j]
        x[i] /= M[i][i]
    return x, steps

def lu_decomp(A, b):
    n = len(A)
    L = [[1.0 if i==j else 0.0 for j in range(n)] for i in range(n)] # identity matrix
    U = [row[:] for row in A]
    steps = []
    for col in range(n):
        if abs(U[col][col]) < 1e-12:
            return None, None, None, None, "Singular matrix."
        for row in range(col+1, n):
            f = U[row][col]/U[col][col]
            L[row][col] = f
            steps.append(f"L[{row+1}][{col+1}] = {f:.5f}")
            for c in range(col, n):
                U[row][c] -= f*U[col][c]
    y = [0.0]*n
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i][j]*y[j]
    x = [0.0]*n
    for i in range(n-1,-1,-1):
        x[i] = y[i]
        for j in range(i+1,n):
            x[i] -= U[i][j]*x[j]
        x[i] /= U[i][i]
    return x, L, U, y, steps

BG       = "#000000"
SURFACE  = "#483D8B"
CARD     = "#0f3460"
BTN_BG   = "#1e1e2e"
BTN_FG   = "#000000"
ACTIVE   = "#313244"
ACCENT   = "#89b4fa"
GREEN    = "#a6e3a1"
RED      = "#f38ba8"
MUTED    = "#6c7086"
WHITE    = "#cdd6f4"
YELLOW   = "#f9e2af"
BORDER   = "#313244"
BLACK    = "#F5FFFA"
NOR      = "#4B0082"

METHOD_PARAMS = {
    "Bisection":    ["f(x)", "a", "b", "Tolerance", "Max Iter"],
    "False Pos.":   ["f(x)", "a", "b", "Tolerance", "Max Iter"],
    "Newton-R":     ["f(x)", "x0", "Tolerance", "Max Iter"],
    "Secant":       ["f(x)", "x0", "x1", "Tolerance", "Max Iter"],
    "Fixed Pt":     ["f(x)", "g(x)", "x0", "Tolerance", "Max Iter"],
}

MAT_METHODS = ["Gaussian", "Partial Pivot", "LU Decomp"]

def show_result_window(title, content):
    win = tk.Toplevel()
    win.title(title)
    win.configure(bg=BG)
    win.geometry("900x600")

    top = tk.Frame(win, bg=BG)
    top.pack(fill='x', padx=12, pady=(12,0))
    tk.Label(top, text=title, bg=BG, fg=ACCENT,
             font=("Consolas", 13, "bold")).pack(side='left')
    tk.Button(top, text="✕ Close", bg=BTN_BG, fg=RED,
              font=("Consolas", 10), relief='flat',
              command=win.destroy).pack(side='right')

    frame = tk.Frame(win, bg=BG)
    frame.pack(fill='both', expand=True, padx=12, pady=8)

    sb = tk.Scrollbar(frame)
    sb.pack(side='right', fill='y')

    txt = tk.Text(frame, bg=SURFACE, fg=WHITE, font=("Consolas", 11),
                  yscrollcommand=sb.set, relief='flat',
                  padx=12, pady=8, wrap='none')
    txt.pack(fill='both', expand=True)
    sb.config(command=txt.yview)

    hb = tk.Scrollbar(win, orient='horizontal', command=txt.xview)
    hb.pack(fill='x', padx=12, pady=(0,8))
    txt.config(xscrollcommand=hb.set)

    txt.tag_config("head",  foreground=ACCENT, font=("Consolas", 11, "bold"))
    txt.tag_config("ok",    foreground=GREEN,  font=("Consolas", 11, "bold"))
    txt.tag_config("err",   foreground=RED,    font=("Consolas", 11, "bold"))
    txt.tag_config("muted", foreground=MUTED)
    txt.tag_config("row",   foreground=WHITE)
    txt.tag_config("step",  foreground=YELLOW)

    for tag, text in content:
        txt.insert('end', text, tag)

    txt.config(state='disabled')

class NumericalCalcApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Numerical Methods Calculator")
        self.root.configure(bg=BG)
        self.root.geometry("800x800")
        self.root.resizable(True, True)

        self.current_tab   = tk.StringVar(value="root")
        self.current_method = tk.StringVar(value="Bisection")
        self.active_param  = tk.StringVar(value="f(x)")
        self.param_values  = {
            "f(x)": tk.StringVar(value=""),
            "g(x)": tk.StringVar(value=""),
            "a":    tk.StringVar(value=""),
            "b":    tk.StringVar(value=""),
            "x0":   tk.StringVar(value=""),
            "x1":   tk.StringVar(value=""),
            "Tolerance": tk.StringVar(value="0.0001"),
            "Max Iter":  tk.StringVar(value="50"),
        }
        self.mat_method = tk.StringVar(value="Gaussian")
        self.mat_n      = tk.IntVar(value=3)
        self.mat_cells  = []
        self.mat_b      = []
        self.active_mat_cell = None


        self._build_ui()

    def _build_ui(self):
        tab_frame = tk.Frame(self.root, bg=BG)
        tab_frame.pack(fill='x', padx=12, pady=(12, 4))

        self.tab_btns = {}
        for label in ["Root Finding Methods", "Matrix Solver Methods"]:
            key = "root" if "Root" in label else "matrix"
            b = tk.Button(tab_frame, text=label, bg=BTN_BG, fg=WHITE,
                          font=("Consolas", 15, "bold"),
                          relief='raised', bd=4, padx=20, pady=10,
                          cursor='hand2',
                          command=lambda k=key: self._switch_tab(k))
            b.pack(side='left', fill='x', expand=True, padx=(0,4))
            self.tab_btns[key] = b

        self.frame_root   = tk.Frame(self.root, bg=BG)
        self.frame_matrix = tk.Frame(self.root, bg=BG)

        self._build_root_tab()
        self._build_matrix_tab()

        self._switch_tab("root")

    def _switch_tab(self, tab):
        self.current_tab.set(tab)
        if tab == "root":
            self.frame_matrix.pack_forget()
            self.frame_root.pack(fill='both', expand=True)
            self.tab_btns["root"].config(bg=CARD, fg=ACCENT)
            self.tab_btns["matrix"].config(bg=BTN_BG, fg=WHITE)
        else:
            self.frame_root.pack_forget()
            self.frame_matrix.pack(fill='both', expand=True)
            self.tab_btns["root"].config(bg=BTN_BG, fg=WHITE)
            self.tab_btns["matrix"].config(bg=CARD, fg=ACCENT)

    def _build_root_tab(self):
        f = self.frame_root

        mrow = tk.Frame(f, bg=BG)
        mrow.pack(fill='y', padx=12, pady=(8, 4))

        self.method_btns = {}
        for m in ["Bisection", "False Pos.", "Newton-R", "Secant", "Fixed Pt"]:
            btn = tk.Button(mrow, text=m, bg=BTN_BG, fg=MUTED,
                            font=("Consolas", 14), relief='raised', padx=14, pady=8,
                            cursor='hand2',
                            command=lambda mm=m: self._set_method(mm))
            btn.pack(side='left', padx=(0, 5))
            self.method_btns[m] = btn

        disp_frame = tk.Frame(f, bg=SURFACE, highlightbackground=BORDER,
                              highlightthickness=1)
        disp_frame.pack(fill='x', padx=12, pady=6)

        self.disp_label = tk.Label(disp_frame, text="f(x)", bg=SURFACE, fg=MUTED,
                                   font=("Consolas", 14),  anchor='w')
        self.disp_label.pack(fill='x', padx=12, pady=(8, 0))

        self.disp_expr = tk.Label(disp_frame, text="", bg=SURFACE, fg=WHITE,
                                  font=("Consolas", 16), anchor='w', pady=8)
        self.disp_expr.pack(fill='x', padx=12, pady=(0, 8))

        self.cards_frame = tk.Frame(f, bg=BG)
        self.cards_frame.pack(fill='x', padx=12, pady=4)

        self.sel_frame = tk.Frame(f, bg=BG)
        self.sel_frame.pack(fill='x', padx=12, pady=4)

        kpad_frame = tk.Frame(f, bg=BG)
        kpad_frame.pack(fill='x', padx=12)

        keys = [
            ("7","7"),("8","8"),("9","9"),("(","("),(")",")"),
            ("4","4"),("5","5"),("6","6"),("×","*"),("÷","/"),
            ("1","1"),("2","2"),("3","3"),("+","+"),(  "−","-"),
            ("0","0"),(".","."),(  "x","x"),("^","^"),("DEL","DEL"),
            ("sin","sin("),("cos","cos("),("tan","tan("),("ln","log("),("sqrt","sqrt("),
            ("exp","exp("),("log","log10("),("pi","pi"),("e","e"),("CLR","CLR"),
        ]

        self.kpad_btns_root = []
        for idx, (lbl, val) in enumerate(keys):
            r, c = divmod(idx, 5)
            is_op  = val in ('*','/','+','-','^','(',')')
            is_del = val in ('DEL','CLR')
            is_fn  = lbl in ('sin','cos','tan','ln','sqrt','exp','log','pi','e')

            fg = ACCENT if is_op else (RED if is_del else (YELLOW if is_fn else WHITE))
            btn = tk.Button(kpad_frame, text=lbl,
                            bg=BTN_BG, fg=fg,
                            font=("Consolas", 14), relief='raised',
                            padx=6, pady=10, cursor='hand2',
                            command=lambda v=val: self._press_key(v))
            btn.grid(row=r, column=c, sticky='nsew', padx=3, pady=3)
            self.kpad_btns_root.append(btn)

        for c in range(5):
            kpad_frame.columnconfigure(c, weight=1)

        tk.Button(f, text="SOLVE", bg=NOR, fg="white",
                  font=("Consolas", 17, "bold"),
                  relief='flat', pady=5, cursor='hand2',
                  command=self._solve_root).pack(fill='x', padx=12, pady=10)

        self._set_method("Bisection")

    def _set_method(self, m):
        self.current_method.set(m)
        params = METHOD_PARAMS[m]
        self.active_param.set(params[0])

        for mm, btn in self.method_btns.items():
            btn.config(bg=CARD if mm == m else BTN_BG,
                       fg=ACCENT if mm == m else MUTED)

        self._rebuild_cards()
        self._rebuild_selector()
        self._update_display()

    def _rebuild_cards(self):
        for w in self.cards_frame.winfo_children():
            w.destroy()

        m = self.current_method.get()
        params = [p for p in METHOD_PARAMS[m] if p not in ("Tolerance","Max Iter")]
        for p in params:
            active = (p == self.active_param.get())
            card = tk.Frame(self.cards_frame, bg=ACTIVE if active else SURFACE,
                            highlightbackground=ACCENT if active else BORDER,
                            highlightthickness=1)
            card.pack(side='left', fill='y', padx=(0,6), pady=2, ipadx=10, ipady=4)
            tk.Label(card, text=p, bg=ACTIVE if active else SURFACE,
                     fg=ACCENT if active else MUTED,
                     font=("Consolas", 11, BOLD)).pack(anchor='w', padx=6, pady=(4,0))
            val = self.param_values[p].get() or "_"
            tk.Label(card, text=val, bg=ACTIVE if active else SURFACE,
                     fg=WHITE, font=("Consolas", 13)).pack(anchor='w', padx=6, pady=(0,4))
            card.bind("<Button-1>", lambda e, pp=p: self._set_active_param(pp))
            for child in card.winfo_children():
                child.bind("<Button-1>", lambda e, pp=p: self._set_active_param(pp))

    def _rebuild_selector(self):
        for w in self.sel_frame.winfo_children():
            w.destroy()
        m = self.current_method.get()
        for p in METHOD_PARAMS[m]:
            active = (p == self.active_param.get())
            btn = tk.Button(self.sel_frame, text=p,
                            bg=ACTIVE if active else BTN_BG,
                            fg=WHITE if active else MUTED,
                            font=("Consolas", 10), relief='raised',
                            padx=18, pady=5, cursor='hand2',
                            command=lambda pp=p: self._set_active_param(pp))
            btn.pack(side='left', padx=(4,3))

    def _set_active_param(self, p):
        self.active_param.set(p)
        self._rebuild_cards()
        self._rebuild_selector()
        self._update_display()

    def _update_display(self):
        p = self.active_param.get()
        self.disp_label.config(text=p)
        val = self.param_values[p].get()
        self.disp_expr.config(text=val + "│")

    def _press_key(self, val):
        p = self.active_param.get()
        cur = self.param_values[p].get()
        if val == "DEL":
            self.param_values[p].set(cur[:-1])
        elif val == "CLR":
            self.param_values[p].set("")
        else:
            self.param_values[p].set(cur + val)
        self._update_display()
        self._rebuild_cards()

    def _solve_root(self):
        m   = self.current_method.get()
        fx  = self.param_values["f(x)"].get().strip()
        gx  = self.param_values["g(x)"].get().strip()
        tol = float(self.param_values["Tolerance"].get() or 0.0001)
        mi  = int(self.param_values["Max Iter"].get() or 50)

        try:
            if m == "Bisection":
                a = float(self.param_values["a"].get())
                b = float(self.param_values["b"].get())
                root, rows, err = bisection(fx, a, b, tol, mi)
                headers = ["Iter","a","b","c","f(c)","Error","Abs.Err","Rel.Err","%Err"]

            elif m == "False Pos.":
                a = float(self.param_values["a"].get())
                b = float(self.param_values["b"].get())
                root, rows, err = false_position(fx, a, b, tol, mi)
                headers = ["Iter","a","b","c","f(c)","|f(c)|","Abs.Err","Rel.Err","%Err"]

            elif m == "Newton-R":
                x0 = float(self.param_values["x0"].get())
                root, rows, err = newton_raphson(fx, x0, tol, mi)
                headers = ["Iter","x_n","f(x_n)","f'(x_n)","x_n+1","Abs.Err","Rel.Err","%Err"]

            elif m == "Secant":
                x0 = float(self.param_values["x0"].get())
                x1 = float(self.param_values["x1"].get())
                root, rows, err = secant(fx, x0, x1, tol, mi)
                headers = ["Iter","x_n-1","x_n","x_n+1","f(x_n+1)","Abs.Err","Rel.Err","%Err"]

            elif m == "Fixed Pt":
                x0 = float(self.param_values["x0"].get())
                root, rows, err = fixed_point(fx, gx, x0, tol, mi)
                headers = ["Iter","x_n","g(x_n)","f(g(x_n))","Abs.Err","Rel.Err","%Err"]

        except Exception as ex:
            content = [("err", f"  ✗ Error: {ex}\n")]
            show_result_window("Error", content)
            return

        if err:
            content = [("err", f"  ✗ {err}\n")]
            show_result_window("Error", content)
            return

        col_w = 16
        header_line = "".join(f"{h:>{col_w}}" for h in headers) + "\n"
        sep = "─" * (col_w * len(headers)) + "\n"

        content = [
            ("head", f"\n  Method: {m}\n"),
            ("head", f"  f(x) = {fx}\n\n"),
            ("muted", sep),
            ("head", header_line),
            ("muted", sep),
        ]
        for row in rows:
            line = "".join(f"{str(v):>{col_w}}" for v in row) + "\n"
            content.append(("row", line))
        content.append(("muted", sep))
        content.append(("ok", f"\n  ✓  ROOT ≈ {root:.10f}   ({len(rows)} iterations)\n"))

        show_result_window(f"Result — {m}", content)

    def _build_matrix_tab(self):
        f = self.frame_matrix

        mrow = tk.Frame(f, bg=BG)
        mrow.pack(fill='x', padx=12, pady=(8, 4))
        self.mat_method_btns = {}
        for mm in MAT_METHODS:
            btn = tk.Button(mrow, text=mm, bg=BTN_BG, fg=MUTED,
                            font=("Consolas", 14), relief='flat', padx=14, pady=8,
                            cursor='hand2',
                            command=lambda m=mm: self._set_mat_method(m))
            btn.pack(side='left', padx=(0,5))
            self.mat_method_btns[mm] = btn
        self._set_mat_method("Gaussian")

        srow = tk.Frame(f, bg=BG)
        srow.pack(fill='x', padx=12, pady=4)

        self.size_btns = {}
        for sz in [2,3,4,5]:
            btn = tk.Button(srow, text=f"{sz}×{sz}",
                            bg=BTN_BG, fg=MUTED,
                            font=("Consolas", 14), relief='flat', padx=12, pady=6,
                            cursor='hand2',
                            command=lambda s=sz: self._set_mat_size(s))
            btn.pack(side='left', padx=(0,4))
            self.size_btns[sz] = btn

        self.mat_area = tk.Frame(f, bg=BG)
        self.mat_area.pack(fill='x', padx=12, pady=8)

        mkpad = tk.Frame(f, bg=BG)
        mkpad.pack(fill='both', padx=12)

        mat_keys = [
            ("7","7"),("8","8"),("9","9"),("-","-"),("DEL","DEL"),
            ("4","4"),("5","5"),("6","6"),("+","+"),("CLR","CLR"),
            ("1","1"),("2","2"),("3","3"),("×","*"),("÷","/"),
            ("0","0"),(".","."),(  "−","-"),("E","e"),(".","."),
        ]
        for idx, (lbl, val) in enumerate(mat_keys):
            r, c = divmod(idx, 5)
            is_del = val in ('DEL','CLR')
            is_op  = val in ('*','/','+','-')
            fg = RED if is_del else (ACCENT if is_op else WHITE)
            tk.Button(mkpad, text=lbl, bg=BTN_BG, fg=fg,
                      font=("Consolas", 12), relief='flat',
                      padx=6, pady=10, cursor='hand2',
                      command=lambda v=val: self._press_mat_key(v)
                      ).grid(row=r, column=c, sticky='nsew', padx=3, pady=3)
        for c in range(5):
            mkpad.columnconfigure(c, weight=1)

        tk.Button(f, text="SOLVE",
                  bg="#1d4ed8", fg="white",
                  font=("Consolas", 16, "bold"),
                  relief='flat', pady=1, cursor='hand2',
                  command=self._solve_matrix).pack(fill='x', padx=12, pady=10)

        self._set_mat_size(3)

    def _set_mat_method(self, m):
        self.mat_method.set(m)
        for mm, btn in self.mat_method_btns.items():
            btn.config(bg=CARD if mm == m else BTN_BG,
                       fg=ACCENT if mm == m else MUTED)

    def _set_mat_size(self, n):
        self.mat_n.set(n)
        for sz, btn in self.size_btns.items():
            btn.config(bg=CARD if sz == n else BTN_BG,
                       fg=YELLOW if sz == n else MUTED)

        self.mat_cells = [[tk.StringVar(value="") for _ in range(n)] for _ in range(n)]
        self.mat_b     = [tk.StringVar(value="") for _ in range(n)]
        self.active_mat_cell = None

        if n == 3:
            defaults_A = [["2","1","-1"],["1","3","2"],["1","-1","4"]]
            defaults_b = ["8","12","4"]
            for i in range(3):
                for j in range(3):
                    self.mat_cells[i][j].set(defaults_A[i][j])
                self.mat_b[i].set(defaults_b[i])

        for w in self.mat_area.winfo_children():
            w.destroy()

        left = tk.Frame(self.mat_area, bg=BG)
        left.pack(side='left')

        self.mat_entry_widgets = []
        for i in range(n):
            row_widgets = []
            for j in range(n):
                e = tk.Entry(left, textvariable=self.mat_cells[i][j],
                             width=7, bg=SURFACE, fg=WHITE,
                             font=("Consolas",12), relief='flat',
                             justify='center',
                             highlightbackground=BORDER, highlightthickness=1,
                             insertbackground=WHITE, readonlybackground=ACTIVE)
                e.grid(row=i+1, column=j, padx=3, pady=3, ipady=6)
                e.bind("<FocusIn>", lambda ev, ii=i, jj=j: self._focus_mat_cell(ii, jj, 'A'))
                row_widgets.append(e)
            self.mat_entry_widgets.append(row_widgets)

        right = tk.Frame(self.mat_area, bg=BG)
        right.pack(side='left', padx=(20,0))
        self.mat_b_widgets = []
        for i in range(n):
            e = tk.Entry(right, textvariable=self.mat_b[i],
                         width=7, bg=SURFACE, fg=WHITE,
                         font=("Consolas",12), relief='flat',
                         justify='center',
                         highlightbackground=BORDER, highlightthickness=1,
                         insertbackground=WHITE)
            e.grid(row=i+1, column=0, padx=3, pady=3, ipady=6)
            e.bind("<FocusIn>", lambda ev, ii=i: self._focus_mat_cell(ii, None, 'b'))
            self.mat_b_widgets.append(e)

    def _focus_mat_cell(self, i, j, which):
        self.active_mat_cell = (which, i, j)

    def _press_mat_key(self, val):
        if not self.active_mat_cell:
            return
        which, i, j = self.active_mat_cell
        if which == 'A':
            var = self.mat_cells[i][j]
            widget = self.mat_entry_widgets[i][j]
        else:
            var = self.mat_b[i]
            widget = self.mat_b_widgets[i]

        cur = var.get()
        if val == "DEL":
            var.set(cur[:-1])
        elif val == "CLR":
            var.set("")
        else:
            var.set("" if cur == "" else cur)
            var.set(var.get() + val)

        widget.icursor('end')

    def _solve_matrix(self):
        n = self.mat_n.get()
        try:
            A = [[float(self.mat_cells[i][j].get()) for j in range(n)] for i in range(n)]
            b = [float(self.mat_b[i].get()) for i in range(n)]
        except ValueError:
            content = [("err", "  ✗ Invalid matrix values. Enter numbers only.\n")]
            show_result_window("Error", content)
            return

        mm = self.mat_method.get()

        try:
            if mm == "Gaussian":
                x, steps = gaussian_elim(A, b, pivot=False)
                if x is None:
                    content = [("err", f"  ✗ {steps}\n")]
                    show_result_window("Error", content)
                    return
                content = _format_gauss_result(A, b, x, steps, mm)

            elif mm == "Partial Pivot":
                x, steps = gaussian_elim(A, b, pivot=True)
                if x is None:
                    content = [("err", f"  ✗ {steps}\n")]
                    show_result_window("Error", content)
                    return
                content = _format_gauss_result(A, b, x, steps, mm)

            elif mm == "LU Decomp":
                x, L, U, y, steps = lu_decomp(A, b)
                if x is None:
                    content = [("err", f"  ✗ {steps}\n")]
                    show_result_window("Error", content)
                    return
                content = _format_lu_result(A, b, x, L, U, y, steps)

        except Exception as ex:
            content = [("err", f"  ✗ {ex}\n")]
            show_result_window("Error", content)
            return

        show_result_window(f"Result — {mm}", content)

def _format_gauss_result(A, b, x, steps, method):
    n = len(A)
    content = [("head", f"\n  Method: {method}\n\n")]
    content.append(("head", "  Original System [A|b]:\n"))
    for i in range(n):
        row_str = "  [ " + "  ".join(f"{v:8.4f}" for v in A[i]) + "  |  " + f"{b[i]:8.4f}" + " ]\n"
        content.append(("row", row_str))
    content.append(("muted", "\n  ── Elimination Steps ──\n"))
    for i, s in enumerate(steps):
        content.append(("step", f"  {i+1}. {s}\n"))
    content.append(("muted", "\n  ── Solution ──\n"))
    for i, xi in enumerate(x):
        content.append(("ok", f"  x{i+1} = {xi:.8f}\n"))
    return content

def _format_lu_result(A, b, x, L, U, y, steps):
    n = len(A)
    content = [("head", "\n  Method: LU Decomposition (Doolittle)\n\n")]

    def mat_block(M, label):
        block = [("head", f"  {label}:\n")]
        for row in M:
            block.append(("row", "  [ " + "  ".join(f"{v:9.5f}" for v in row) + " ]\n"))
        return block

    content += mat_block(L, "L Matrix")
    content.append(("muted", "\n"))
    content += mat_block(U, "U Matrix")
    content.append(("muted", "\n  ── Forward Sub [L]{y} = {b} ──\n"))
    for i, yi in enumerate(y):
        content.append(("step", f"  y{i+1} = {yi:.8f}\n"))
    content.append(("muted", "\n  ── Back Sub [U]{x} = {y} ──\n"))
    for i, xi in enumerate(x):
        content.append(("ok", f"  x{i+1} = {xi:.8f}\n"))
    return content

if __name__ == "__main__":
    root = tk.Tk()

    root.title("Numerical Methods Calculator")

    root.update_idletasks()
    w, h = 780, 820
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    app = NumericalCalcApp(root)
    root.mainloop()