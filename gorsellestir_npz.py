import sys
import os
import glob
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt

class NPZViewerApp:
    def __init__(self, root, initial_dir):
        self.root = root
        self.root.title("Gelişmiş NPZ Görüntüleyici ve Gezgini")
        # Pencere boyutunu daha büyük ayarlayalım ki 3 panel sığsın
        self.root.geometry("1600x850")
        
        self.current_dir = initial_dir
        self.files = []
        
        # Ana Bölme (Sol: Liste, Orta: RGB Önizleme, Sağ: Bant Grid)
        self.paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)
        
        # --------- 1. SOL PANEL (Arama ve Seçim) ---------
        self.left_frame = ttk.Frame(self.paned_window, width=280)
        self.paned_window.add(self.left_frame, weight=1)
        
        self.btn_frame = ttk.Frame(self.left_frame)
        self.btn_frame.pack(fill=tk.X, padx=5, pady=5)
        self.btn_select_dir = ttk.Button(self.btn_frame, text="Klasör Seç / Değiştir", command=self.select_directory)
        self.btn_select_dir.pack(fill=tk.X, expand=True)
        
        self.info_label = ttk.Label(self.left_frame, text="Bekleniyor...", wraplength=250, font=("Segoe UI", 10))
        self.info_label.pack(fill=tk.X, padx=8, pady=8)
        
        self.nav_frame = ttk.Frame(self.left_frame)
        self.nav_frame.pack(fill=tk.X, padx=5, pady=5)
        self.btn_prev = ttk.Button(self.nav_frame, text="< Önceki", command=self.prev_file)
        self.btn_prev.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self.btn_next = ttk.Button(self.nav_frame, text="Sonraki >", command=self.next_file)
        self.btn_next.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        self.listbox_frame = ttk.Frame(self.left_frame)
        self.listbox_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.scrollbar = ttk.Scrollbar(self.listbox_frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.listbox = tk.Listbox(self.listbox_frame, yscrollcommand=self.scrollbar.set, font=("Segoe UI", 10))
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=self.listbox.yview)
        self.listbox.bind("<<ListboxSelect>>", self.on_select)
        
        # --------- 2. ORTA PANEL (Önizleme - RGB Kompozit) ---------
        self.middle_frame = ttk.Frame(self.paned_window, width=600)
        self.paned_window.add(self.middle_frame, weight=3)
        
        self.fig_rgb = plt.Figure(figsize=(6, 6))
        self.canvas_rgb = FigureCanvasTkAgg(self.fig_rgb, master=self.middle_frame)
        self.canvas_rgb.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar_rgb = NavigationToolbar2Tk(self.canvas_rgb, self.middle_frame)
        
        # --------- 3. SAĞ PANEL (Tüm Bantlar Grid) ---------
        self.right_frame = ttk.Frame(self.paned_window, width=650)
        self.paned_window.add(self.right_frame, weight=4)
        
        self.fig_bands = plt.Figure(figsize=(8, 8))
        self.canvas_bands = FigureCanvasTkAgg(self.fig_bands, master=self.right_frame)
        self.canvas_bands.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar_bands = NavigationToolbar2Tk(self.canvas_bands, self.right_frame)
        
        # Başlangıç Yüklemesi
        if self.current_dir and os.path.exists(self.current_dir):
            if os.path.isfile(self.current_dir):
                self.load_directory(os.path.dirname(self.current_dir), select_file=self.current_dir)
            else:
                self.load_directory(self.current_dir)

    def select_directory(self):
        d = filedialog.askdirectory(initialdir=self.current_dir, title="NPZ Klasörünü Seçin")
        if d:
            self.load_directory(d)
            
    def load_directory(self, d, select_file=None):
        self.current_dir = d
        self.files = sorted(glob.glob(os.path.join(d, "*.npz")))
        
        self.listbox.delete(0, tk.END)
        for f in self.files:
            self.listbox.insert(tk.END, os.path.basename(f))
        
        if self.files:
            target_idx = 0
            if select_file and select_file in self.files:
                target_idx = self.files.index(select_file)
                
            self.listbox.selection_set(target_idx)
            self.listbox.see(target_idx)
            self.visualize(self.files[target_idx])
        else:
            self.info_label.config(text=f"Klasör: {d}\n\nHiç .npz dosyası bulunamadı!")
            self.fig_bands.clear()
            self.fig_rgb.clear()
            self.canvas_bands.draw()
            self.canvas_rgb.draw()
            
    def prev_file(self):
        sel = self.listbox.curselection()
        if sel:
            idx = sel[0]
            if idx > 0:
                self.listbox.selection_clear(0, tk.END)
                self.listbox.selection_set(idx - 1)
                self.listbox.see(idx - 1)
                self.visualize(self.files[idx - 1])

    def next_file(self):
        sel = self.listbox.curselection()
        if sel:
            idx = sel[0]
            if idx < len(self.files) - 1:
                self.listbox.selection_clear(0, tk.END)
                self.listbox.selection_set(idx + 1)
                self.listbox.see(idx + 1)
                self.visualize(self.files[idx + 1])
                
    def on_select(self, event):
        selection = self.listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        self.visualize(self.files[idx])
        
    def visualize(self, file_path):
        self.fig_bands.clear()
        self.fig_rgb.clear()
        
        try:
            data = np.load(file_path)
            keys = list(data.keys())
            key = 'image' if 'image' in keys else keys[0]
            img = data[key].astype(np.float32) # Güvenlik için float32 formatı
            
            # Bilgi Metni
            info_text = f"AÇIK:\n{os.path.basename(file_path)}\n\n"
            info_text += f"Bant Sayısı / Boyut:\n{img.shape}\n\n"
            info_text += f"Tip: {img.dtype}\n\n"
            info_text += f"Min Değer: {np.min(img):.3f}\n"
            info_text += f"Max Değer: {np.max(img):.3f}"
            self.info_label.config(text=info_text)
            
            # --- 1) ORTA PANEL (RGB ÇİZİMİ) ---
            num_bands = img.shape[0] if (len(img.shape) == 3 and img.shape[0] < img.shape[1]) else 1
            
            ax_rgb = self.fig_rgb.add_subplot(111)
            
            if num_bands >= 3:
                rgb = np.stack([img[0], img[1], img[2]], axis=-1)
                # Görüntüyü canlandırmak için Histogram Germe (Percentile 2-98 Stretch)
                rgb_min = np.percentile(rgb, 2, axis=(0,1), keepdims=True)
                rgb_max = np.percentile(rgb, 98, axis=(0,1), keepdims=True)
                diff = rgb_max - rgb_min
                diff[diff == 0] = 1e-8 # Sıfıra bölmeyi engelle
                
                rgb_norm = (rgb - rgb_min) / diff
                rgb_norm = np.clip(rgb_norm, 0, 1)
                
                ax_rgb.imshow(rgb_norm)
                ax_rgb.set_title("Orijinal Renkli Özeti (RGB Bant 1,2,3)", fontsize=12, fontweight='bold', pad=10)
            else:
                ax_rgb.imshow(img.squeeze() if num_bands == 1 else img[0], cmap='gray')
                ax_rgb.set_title("Önizleme (İlk Bant)", fontsize=12, fontweight='bold', pad=10)
                
            ax_rgb.axis('off')
            
            # --- 2) SAĞ PANEL BANTLAR (Grid Çizimi) ---
            if len(img.shape) == 3 and img.shape[0] < img.shape[1]:
                cols = 4 if num_bands >= 4 else num_bands
                rows = int(np.ceil(num_bands / cols))
                
                axes = self.fig_bands.subplots(rows, cols)
                if num_bands == 1:
                    axes_flat = [axes]
                else:
                    axes_flat = axes.flatten()
                
                for i in range(rows * cols):
                    ax = axes_flat[i]
                    if i < num_bands:
                        im = ax.imshow(img[i], cmap='viridis')
                        ax.set_title(f"Bant {i+1}", fontsize=10)
                        # Daha temiz görünmesi için renk çubuğunu kaldırdık
                        # self.fig_bands.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    ax.axis('off')
                    
            else:
                ax = self.fig_bands.add_subplot(111)
                ax.imshow(img.squeeze(), cmap='gray')
                ax.set_title(os.path.basename(file_path))
                ax.axis('off')
                
            self.fig_bands.tight_layout()
            self.fig_rgb.tight_layout()
            
            self.canvas_bands.draw()
            self.canvas_rgb.draw()
            
        except Exception as e:
            self.info_label.config(text=f"Hata oluştu:\n{e}")

if __name__ == '__main__':
    root = tk.Tk()
    
    # Argüman geldiyse oradan başla, gelmediyse ana dizinden
    initial = os.getcwd()
    if len(sys.argv) > 1:
        initial = sys.argv[1]
        
    app = NPZViewerApp(root, initial)
    
    # Windowsu en üstte parlatıp odaklanması için küçük bir eklenti
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    
    root.mainloop()

