import sys
import os
import glob
import shutil
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
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
        self.root.bind("<Configure>", self.on_window_resize)
        
        self.current_dir = initial_dir
        self.files = []

        self.current_file_label = ttk.Label(
            self.root,
            text="Se\u00e7ili dosya: -",
            anchor=tk.W,
            justify=tk.LEFT,
            font=("Segoe UI", 10, "bold"),
            wraplength=1500,
        )
        self.current_file_label.pack(fill=tk.X, padx=8, pady=(8, 0))
        
        # Ana Bölme (Sol: Liste, Orta: RGB Önizleme, Sağ: Bant Grid)
        self.paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)
        
        # --------- 1. SOL PANEL (Arama ve Seçim) ---------
        self.left_frame = ttk.Frame(self.paned_window, width=380)
        self.paned_window.add(self.left_frame, weight=1)
        
        self.btn_frame = ttk.Frame(self.left_frame)
        self.btn_frame.pack(fill=tk.X, padx=5, pady=5)
        self.btn_select_dir = ttk.Button(self.btn_frame, text="Klasör Seç / Değiştir", command=self.select_directory)
        self.btn_select_dir.pack(fill=tk.X, expand=True)
        
        self.info_label = ttk.Label(self.left_frame, text="Bekleniyor...", wraplength=350, font=("Segoe UI", 10))
        self.info_label.pack(fill=tk.X, padx=8, pady=8)
        
        self.nav_frame = ttk.Frame(self.left_frame)
        self.nav_frame.pack(fill=tk.X, padx=5, pady=5)
        self.btn_prev = ttk.Button(self.nav_frame, text="< Önceki", command=self.prev_file)
        self.btn_prev.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self.btn_next = ttk.Button(self.nav_frame, text="Sonraki >", command=self.next_file)
        self.btn_next.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        self.btn_delete = ttk.Button(
            self.left_frame,
            text="Seçili Dosyayı Sil (Del)",
            command=self.delete_current_file,
        )
        self.btn_delete.pack(fill=tk.X, padx=5, pady=(0, 5))

        self.btn_move = ttk.Button(
            self.left_frame,
            text="Karşı Sınıfa Taşı (M)",
            command=self.move_current_file_to_other_class,
        )
        self.btn_move.pack(fill=tk.X, padx=5, pady=(0, 5))

        self.listbox_frame = ttk.Frame(self.left_frame)
        self.listbox_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.scrollbar = ttk.Scrollbar(self.listbox_frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.scrollbar_x = ttk.Scrollbar(self.listbox_frame, orient=tk.HORIZONTAL)
        self.scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.listbox = tk.Listbox(
            self.listbox_frame,
            yscrollcommand=self.scrollbar.set,
            xscrollcommand=self.scrollbar_x.set,
            font=("Consolas", 10),
        )
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=self.listbox.yview)
        self.scrollbar_x.config(command=self.listbox.xview)
        self.listbox.bind("<<ListboxSelect>>", self.on_select)
        self.listbox.bind("<Delete>", self.delete_current_file)
        self.root.bind("<Delete>", self.delete_current_file)
        self.root.bind("<Key-m>", self.move_current_file_to_other_class)
        self.root.bind("<Key-M>", self.move_current_file_to_other_class)
        
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
            
    def on_window_resize(self, event=None):
        width = max(self.root.winfo_width() - 40, 300)
        self.current_file_label.config(wraplength=width)

        left_width = max(self.left_frame.winfo_width() - 30, 180)
        self.info_label.config(wraplength=left_width)

    def load_directory(self, d, select_file=None):
        self.current_dir = d
        self.files = sorted(glob.glob(os.path.join(d, "*.npz")))
        self.update_action_buttons()
        
        self.listbox.delete(0, tk.END)
        for f in self.files:
            self.listbox.insert(tk.END, os.path.basename(f))
        
        if self.files:
            target_idx = 0
            if select_file and select_file in self.files:
                target_idx = self.files.index(select_file)
                
            self.select_index(target_idx)
        else:
            self.show_empty_directory_message()
            
    def prev_file(self):
        sel = self.listbox.curselection()
        if sel:
            idx = sel[0]
            if idx > 0:
                self.select_index(idx - 1)

    def next_file(self):
        sel = self.listbox.curselection()
        if sel:
            idx = sel[0]
            if idx < len(self.files) - 1:
                self.select_index(idx + 1)
                
    def on_select(self, event):
        selection = self.listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        self.visualize(self.files[idx])

    def select_index(self, idx):
        if not self.files:
            self.show_empty_directory_message()
            return

        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(idx)
        self.listbox.activate(idx)
        self.listbox.see(idx)
        self.visualize(self.files[idx])

    def clear_canvases(self):
        self.fig_bands.clear()
        self.fig_rgb.clear()
        self.canvas_bands.draw()
        self.canvas_rgb.draw()

    def show_empty_directory_message(self):
        self.info_label.config(text=f"Klasör: {self.current_dir}\n\nHiç .npz dosyası bulunamadı!")
        self.current_file_label.config(text="Se\u00e7ili dosya: -")
        self.clear_canvases()
        self.update_action_buttons()

    def update_action_buttons(self):
        has_files = bool(self.files)
        self.btn_delete.config(state=tk.NORMAL if has_files else tk.DISABLED)

        target_dir, action_text = self.get_other_class_target()
        if has_files and target_dir:
            self.btn_move.config(state=tk.NORMAL, text=f"{action_text} (M)")
        else:
            self.btn_move.config(state=tk.DISABLED, text="Karşı Sınıfa Taşı (M)")

    def get_other_class_target(self):
        current_name = os.path.basename(os.path.normpath(self.current_dir)).lower()
        if current_name == "positive":
            target_name = "negative"
            action_text = "Negatife Taşı"
        elif current_name == "negative":
            target_name = "positive"
            action_text = "Pozitife Taşı"
        else:
            return None, None

        parent_dir = os.path.dirname(os.path.normpath(self.current_dir))
        for entry in os.scandir(parent_dir):
            if entry.is_dir() and entry.name.lower() == target_name:
                return entry.path, action_text
        return None, None

    def remove_selected_file_from_list(self, idx):
        self.files.pop(idx)
        self.listbox.delete(idx)

        if self.files:
            next_idx = min(idx, len(self.files) - 1)
            self.select_index(next_idx)
        else:
            self.show_empty_directory_message()
            self.listbox.selection_clear(0, tk.END)

        self.update_action_buttons()

    def delete_current_file(self, event=None):
        selection = self.listbox.curselection()
        if not selection:
            return

        idx = selection[0]
        file_path = self.files[idx]
        file_name = os.path.basename(file_path)

        confirm = messagebox.askyesno(
            "NPZ Dosyasını Sil",
            f"{file_name} dosyası silinsin mi?\n\nBu işlem geri alınamaz.",
            icon=messagebox.WARNING,
        )
        if not confirm:
            return

        try:
            os.remove(file_path)
        except FileNotFoundError:
            messagebox.showwarning("Dosya Bulunamadı", f"{file_name} zaten silinmiş görünüyor.")
        except OSError as exc:
            messagebox.showerror("Silme Hatası", f"{file_name} silinemedi.\n\n{exc}")
            return

        self.remove_selected_file_from_list(idx)

    def move_current_file_to_other_class(self, event=None):
        selection = self.listbox.curselection()
        if not selection:
            return

        target_dir, action_text = self.get_other_class_target()
        if not target_dir:
            messagebox.showwarning(
                "Hedef Klasör Bulunamadı",
                "Bu klasör Positive/Negative yapısında görünmüyor ya da karşı klasör bulunamadı.",
            )
            return

        idx = selection[0]
        file_path = self.files[idx]
        file_name = os.path.basename(file_path)
        target_path = os.path.join(target_dir, file_name)

        confirm = messagebox.askyesno(
            "Sınıf Değiştir",
            f"{file_name} dosyası\n\n{action_text.lower()}?\n\nHedef: {target_dir}",
            icon=messagebox.QUESTION,
        )
        if not confirm:
            return

        if os.path.exists(target_path):
            messagebox.showerror(
                "Hedefte Dosya Var",
                f"Hedef klasörde aynı isimli dosya zaten var:\n\n{target_path}",
            )
            return

        try:
            shutil.move(file_path, target_path)
        except OSError as exc:
            messagebox.showerror("Taşıma Hatası", f"{file_name} taşınamadı.\n\n{exc}")
            return

        self.remove_selected_file_from_list(idx)
        
    def visualize(self, file_path):
        self.fig_bands.clear()
        self.fig_rgb.clear()
        
        try:
            data = np.load(file_path)
            keys = list(data.keys())
            key = 'image' if 'image' in keys else keys[0]
            img = data[key].astype(np.float32) # Güvenlik için float32 formatı
            file_name = os.path.basename(file_path)
            self.current_file_label.config(text=f"Seçili dosya: {file_name}")

            # Bilgi Metni
            info_text = f"AÇIK:\n{file_name}\n\n"
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
                
                # İsimlerin iyice sadeleşmesi için sadece İngilizce etiketler
                band_isimleri = {
                    0: "1: Red",
                    1: "2: Green",
                    2: "3: Blue",
                    3: "4: DSM",
                    4: "5: DTM",
                    5: "6: SVF",
                    6: "7: Pos_Openness",
                    7: "8: Neg_Openness",
                    8: "9: LRM",
                    9: "10: Slope",
                    10: "11: nDSM",
                    11: "12: TPI",
                }

                for i in range(rows * cols):
                    ax = axes_flat[i]
                    if i < num_bands:
                        im = ax.imshow(img[i], cmap='viridis')
                        b_isim = band_isimleri.get(i, f"Bant {i+1}")
                        # Kalın fontu kaldırıp boyutu çok daha küçülttük (8)
                        ax.set_title(b_isim, fontsize=8)
                    ax.axis('off')
                    
            else:
                ax = self.fig_bands.add_subplot(111)
                ax.imshow(img.squeeze(), cmap='gray')
                ax.set_title(os.path.basename(file_path))
                ax.axis('off')
                
            # Grafikler arasına bolca nefes alacak boşluk koyuyoruz (h_pad yatay/dikey boşluk)
            self.fig_bands.tight_layout(pad=1.5, h_pad=2.5, w_pad=1.0)
            self.fig_rgb.tight_layout(pad=1.5)
            
            self.canvas_bands.draw()
            self.canvas_rgb.draw()
            
        except Exception as e:
            self.current_file_label.config(text="Seçili dosya: -")
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

