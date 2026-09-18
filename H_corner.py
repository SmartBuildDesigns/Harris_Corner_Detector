import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Ellipse
import tkinter as tk
from tkinter import filedialog

def procesar_matrices_harris(imagen_original, tamano_ventana=3, k=0.04):
    """
    Calcula toda la matemática pesada del detector de Harris una sola vez.
    Retorna R, la máscara de máximos locales y los tensores.
    """
    if len(imagen_original.shape) == 3:
        imagen_gris = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY)
    else:
        imagen_gris = imagen_original.copy()
        
    img = np.float32(imagen_gris)

    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    Ixx, Iyy, Ixy = Ix**2, Iy**2, Ix*Iy

    Sxx = cv2.GaussianBlur(Ixx, (tamano_ventana, tamano_ventana), 0)
    Syy = cv2.GaussianBlur(Iyy, (tamano_ventana, tamano_ventana), 0)
    Sxy = cv2.GaussianBlur(Ixy, (tamano_ventana, tamano_ventana), 0)

    det_M = (Sxx * Syy) - (Sxy ** 2)
    trace_M = Sxx + Syy
    R = det_M - k * (trace_M ** 2)

    # Precomputar la Supresión de No Máximos (NMS)
    R_dilatado = cv2.dilate(R, None)
    es_maximo_local = (R == R_dilatado)
    
    return R, es_maximo_local, Sxx, Syy, Sxy

class DashboardHarris:
    def __init__(self, imagen, R, es_maximo_local, Sxx, Syy, Sxy, umbral_inicial=0.01):
        self.R = R
        self.es_maximo_local = es_maximo_local
        self.Sxx, self.Syy, self.Sxy = Sxx, Syy, Sxy
        self.idx_actual = 0
        self.puntos = []
        self.num_puntos = 0
        
        if len(imagen.shape) == 3:
            self.img_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        else:
            self.img_rgb = cv2.cvtColor(imagen, cv2.COLOR_GRAY2RGB)

        self.fig = plt.figure(figsize=(15, 7))
        self.fig.canvas.manager.set_window_title('Dashboard - Harris Corner Detector')
        self.ax_img = self.fig.add_subplot(1, 3, 1)
        self.ax_3d = self.fig.add_subplot(1, 3, 2, projection='3d')
        self.ax_ell = self.fig.add_subplot(1, 3, 3)
        plt.subplots_adjust(top=0.85, bottom=0.25, wspace=0.3)
        
        # --- WIDGETS ---
        ax_slider = plt.axes([0.25, 0.12, 0.5, 0.03])
        self.slider_umbral = Slider(
            ax_slider, 'Umbral Relativo', 0.001, 0.2, 
            valinit=umbral_inicial, valstep=0.001
        )
        self.slider_umbral.on_changed(self.al_cambiar_umbral)

        ax_btn_prev = plt.axes([0.35, 0.04, 0.1, 0.05])
        ax_btn_next = plt.axes([0.55, 0.04, 0.1, 0.05])
        self.btn_prev = Button(ax_btn_prev, '<< Anterior')
        self.btn_next = Button(ax_btn_next, 'Siguiente >>')
        
        self.btn_prev.on_clicked(self.ir_anterior)
        self.btn_next.on_clicked(self.ir_siguiente)
        
        self.al_cambiar_umbral(umbral_inicial)
        plt.show()

    def al_cambiar_umbral(self, val):
        umbral_absoluto = val * self.R.max()
        mapa_umbralizado = (self.R > umbral_absoluto)
        
        esquinas_finales = mapa_umbralizado & self.es_maximo_local
        self.puntos = np.argwhere(esquinas_finales > 0)
        self.num_puntos = len(self.puntos)
        
        # ACTUALIZAR EL TÍTULO PRINCIPAL CON EL TOTAL DE ESQUINAS
        self.fig.suptitle(f'Detector de Esquinas de Harris | Total detectadas: {self.num_puntos}', 
                          fontsize=16, fontweight='bold')
        
        self.idx_actual = 0
        self.actualizar_dashboard()

    def actualizar_dashboard(self):
        self.ax_img.clear()
        self.ax_3d.clear()
        self.ax_ell.clear()

        if self.num_puntos == 0:
            self.ax_img.imshow(self.img_rgb)
            self.ax_img.set_title("Sube o baja el umbral para encontrar esquinas")
            self.ax_img.axis('off')
            self.fig.canvas.draw_idle()
            return

        y, x = self.puntos[self.idx_actual]
        
        # 1. ACTUALIZAR IMAGEN
        self.ax_img.imshow(self.img_rgb)
        y_all, x_all = zip(*self.puntos)
        self.ax_img.scatter(x_all, y_all, color='red', s=15, alpha=0.5)
        self.ax_img.scatter([x], [y], color='lime', s=100, edgecolor='black', zorder=5)
        
        self.ax_img.set_title(f"Visualizando esquina {self.idx_actual + 1}\n(x: {x}, y: {y})")
        self.ax_img.axis('off')

        # 2. ACTUALIZAR SUPERFICIE 3D E(u,v)
        M_xx, M_yy, M_xy = self.Sxx[y, x], self.Syy[y, x], self.Sxy[y, x]
        u, v = np.linspace(-5, 5, 20), np.linspace(-5, 5, 20)
        U, V = np.meshgrid(u, v)
        E = (M_xx * U**2) + (2 * M_xy * U * V) + (M_yy * V**2)
        
        self.ax_3d.plot_surface(U, V, E, cmap='inferno', edgecolor='none')
        self.ax_3d.set_title('Superficie Cuadrática 3D')
        self.ax_3d.set_xlabel('u')
        self.ax_3d.set_ylabel('v')
        self.ax_3d.set_xticklabels([])
        self.ax_3d.set_yticklabels([])
        self.ax_3d.set_zticklabels([])

        # 3. ACTUALIZAR ELIPSE 2D Y VECTORES
        M = np.array([[M_xx, M_xy], [M_xy, M_yy]])
        eigenvalores, eigenvectores = np.linalg.eigh(M)
        
        l1, l2 = max(eigenvalores[0], 1e-10), max(eigenvalores[1], 1e-10)
        
        w = 2 / np.sqrt(l1)
        h = 2 / np.sqrt(l2)
        escala = 2.0 / max(w, h)
        w_norm, h_norm = w * escala, h * escala
        
        angulo_grados = np.degrees(np.arctan2(eigenvectores[1, 0], eigenvectores[0, 0]))
        
        elipse = Ellipse(xy=(0, 0), width=w_norm, height=h_norm, angle=angulo_grados, 
                         edgecolor='blue', fc='cyan', alpha=0.3, lw=2)
        self.ax_ell.add_patch(elipse)
        
        self.ax_ell.quiver(0, 0, eigenvectores[0,0], eigenvectores[1,0], angles='xy', scale_units='xy', scale=1, color='red', label=f'v1 (λ1={l1:.1e})')
        self.ax_ell.quiver(0, 0, eigenvectores[0,1], eigenvectores[1,1], angles='xy', scale_units='xy', scale=1, color='green', label=f'v2 (λ2={l2:.1e})')
        
        rango = 1.5
        self.ax_ell.set_xlim(-rango, rango)
        self.ax_ell.set_ylim(-rango, rango)
        self.ax_ell.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        self.ax_ell.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        self.ax_ell.set_aspect('equal')
        self.ax_ell.set_title('Eigenvectores y Tensor Estructural')
        self.ax_ell.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25))
        
        info_str = f"Respuesta (R): {self.R[y, x]:.1e}\nMatriz M:\n[[{M_xx:.1e}, {M_xy:.1e}]\n [{M_xy:.1e}, {M_yy:.1e}]]"
        self.ax_ell.text(0.05, 0.95, info_str, transform=self.ax_ell.transAxes, 
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        self.fig.canvas.draw_idle()

    def ir_siguiente(self, event):
        if self.num_puntos > 0:
            self.idx_actual = (self.idx_actual + 1) % self.num_puntos
            self.actualizar_dashboard()

    def ir_anterior(self, event):
        if self.num_puntos > 0:
            self.idx_actual = (self.idx_actual - 1) % self.num_puntos
            self.actualizar_dashboard()

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    ruta_imagen = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png *.bmp *.tif")]
    )

    if ruta_imagen:
        imagen = cv2.imread(ruta_imagen)
        if imagen is not None:
            print("Calculando derivadas y tensores estructurales...")
            R, mask_nms, Sxx, Syy, Sxy = procesar_matrices_harris(imagen)
            print("Iniciando Dashboard...")
            app = DashboardHarris(imagen, R, mask_nms, Sxx, Syy, Sxy)
        else:
            print("Error: No se pudo cargar la imagen.")