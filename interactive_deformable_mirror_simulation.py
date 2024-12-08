import numpy as np
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLineEdit, 
    QLabel, QSlider, QHBoxLayout, QVBoxLayout, QGroupBox, 
    QFormLayout, QComboBox, QCheckBox, QPushButton, 
    QDoubleSpinBox, QScrollArea
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
from numba import njit
from numpy.fft import fft2, ifftshift, fftshift

@njit
def factorial_jit(n):
    res = 1
    for i in range(2, n+1):
        res *= i
    return res

@njit
def zernike_radial(n, m, rho):
    R = np.zeros_like(rho)
    for k in range((n - abs(m)) // 2 + 1):
        nf = factorial_jit(n - k)
        kf = factorial_jit(k)
        af = factorial_jit((n + abs(m)) // 2 - k)
        bf = factorial_jit((n - abs(m)) // 2 - k)
        c = ((-1)**k * nf) / (kf * af * bf)
        R += c * rho**(n - 2*k)
    return R

@njit
def compute_zernike(n, m, rho, theta):
    R = zernike_radial(n, abs(m), rho)
    if m > 0:
        return R * np.cos(m * theta)
    elif m < 0:
        return R * np.sin(abs(m) * theta)
    else:
        return R

class ZernikeGenerator:
    def __init__(self, zernike_indices, R, Theta, mask):
        self.zernike_indices = zernike_indices
        self.num_zernike = len(zernike_indices)
        self.zernike_modes = self.precompute_zernike_modes(R, Theta, mask)

    def precompute_zernike_modes(self, R, Theta, mask):
        modes = np.zeros((self.num_zernike, R.shape[0], R.shape[1]))
        for i, (n, m) in enumerate(self.zernike_indices):
            Z = compute_zernike(n, m, R, Theta)
            Z *= mask
            modes[i] = Z
        return modes

    def generate_wavefront(self, coefficients):
        return np.tensordot(coefficients, self.zernike_modes, axes=(0, 0))

class DeformableMirror:
    def __init__(self, actuator_positions, sigma, X, Y, mask):
        self.actuator_positions = actuator_positions
        self.sigma = sigma
        self.X = X
        self.Y = Y
        self.mask = mask
        self.influence_matrix = self.precompute_influence()

    def precompute_influence(self):
        num_actuators = len(self.actuator_positions)
        mask_flat = self.mask.flatten()
        Xf = self.X.flatten()
        Yf = self.Y.flatten()

        infl = np.zeros((np.sum(mask_flat), num_actuators))
        for j, (xc, yc) in enumerate(self.actuator_positions):
            infl[:, j] = np.exp(-((Xf[mask_flat] - xc)**2 + (Yf[mask_flat] - yc)**2)/(2*self.sigma**2))
        return infl

    def deformable_mirror(self, coefficients):
        shape_flat = self.influence_matrix @ coefficients
        dm_shape = np.zeros_like(self.X)
        dm_shape[self.mask] = shape_flat
        return dm_shape

class PSFSimulator:
    def __init__(self, wavelength):
        self.wavelength = wavelength

    def crop_center(self, psf, crop_size=64):
        """Crop the center of the PSF to the specified size."""
        center = psf.shape[0] // 2
        half_crop = crop_size // 2
        return psf[center - half_crop:center + half_crop,
                   center - half_crop:center + half_crop]

    def simulate_psf(self, wavefront, mask, grid_size=100, padding_factor=2, crop_size=64):
        k = 2 * np.pi / self.wavelength
        complex_wavefront = np.exp(1j * k * wavefront) * mask

        padded_size = grid_size * padding_factor
        padded_wavefront = np.zeros((padded_size, padded_size), dtype=np.complex128)
        offset = (padded_size - grid_size) // 2
        padded_wavefront[offset:offset + grid_size, offset:offset + grid_size] = complex_wavefront

        fft_wavefront = fftshift(fft2(ifftshift(padded_wavefront)))
        psf = np.abs(fft_wavefront)**2
        psf /= psf.sum()

        return self.crop_center(psf, crop_size=crop_size)

def generate_actuator_positions(arrangement_type, num_actuators_x, num_actuators_y, extend_outside):
    if arrangement_type == "grid":
        if extend_outside:
            min_val, max_val = -1.2, 1.2
        else:
            min_val, max_val = -1, 1
        actuator_x_positions = np.linspace(min_val, max_val, num_actuators_x)
        actuator_y_positions = np.linspace(min_val, max_val, num_actuators_y)
        actuator_positions = [
            (xx, yy) for xx in actuator_x_positions for yy in actuator_y_positions
        ]
    elif arrangement_type == "circular":
        if extend_outside:
            max_radius = 1.2
        else:
            max_radius = 1.0
        rings = np.linspace(0.0, max_radius, num_actuators_y)
        angles = np.linspace(0, 2*np.pi, num_actuators_x, endpoint=False)
        actuator_positions = []
        for r in rings:
            for a in angles:
                actuator_positions.append((r*np.cos(a), r*np.sin(a)))
    else:
        raise ValueError("Unknown arrangement_type.")
    return actuator_positions

class CustomScrollArea(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)

    def wheelEvent(self, event):
        # Handle wheel event for scrolling
        delta = event.angleDelta().y()
        if delta > 0:
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - 20)
        elif delta < 0:
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() + 20)
        event.accept()  # Prevent propagation to child widgets

class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(8, 5))
        super().__init__(self.fig)
        self.ax = self.fig.subplots(2, 3)
        self.fig.tight_layout()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.fixed_grid_size = 100  # Fixed internal grid size
        self.padding_factor = 4
        self.crop_size = 128  # Fixed crop size
        self.wavelength = 0.8
        self.sigma = 0.2
        self.arrangement_type = "grid"
        self.extend_outside = False
        self.num_actuators_x = 5
        self.num_actuators_y = 5

        # Increase the number of Zernike polynomials (e.g., n up to 7 for 36 modes)
        self.zernike_indices = []
        for n in range(8):
            for m in range(-n, n+1, 2):
                self.zernike_indices.append((n, m))

        self.build_arrays()
        self.num_zernike = len(self.zernike_indices)
        self.coefficients = np.zeros(self.num_zernike)

        widget = QWidget()
        self.setCentralWidget(widget)
        main_layout = QHBoxLayout(widget)

        self.canvas = MplCanvas()
        main_layout.addWidget(self.canvas)

        control_layout = QVBoxLayout()

        # Parameters group
        param_group = QGroupBox("Parameters")
        form_layout = QFormLayout()
        self.wavelength_input = QLineEdit(str(self.wavelength))
        self.sigma_input = QLineEdit(str(self.sigma))
        self.arrangement_input = QComboBox()
        self.arrangement_input.addItems(["grid", "circular"])
        self.arrangement_input.setCurrentText(self.arrangement_type)
        self.actuators_x_input = QLineEdit(str(self.num_actuators_x))
        self.actuators_y_input = QLineEdit(str(self.num_actuators_y))

        form_layout.addRow(QLabel("Wavelength"), self.wavelength_input)
        form_layout.addRow(QLabel("Sigma"), self.sigma_input)
        form_layout.addRow(QLabel("Arrangement"), self.arrangement_input)
        form_layout.addRow(QLabel("Actuators X"), self.actuators_x_input)
        form_layout.addRow(QLabel("Actuators Y"), self.actuators_y_input)
        param_group.setLayout(form_layout)
        control_layout.addWidget(param_group)

        # Connect parameters to auto-apply
        self.wavelength_input.textChanged.connect(self.apply_settings)
        self.sigma_input.textChanged.connect(self.apply_settings)
        self.arrangement_input.currentIndexChanged.connect(self.apply_settings)
        self.actuators_x_input.textChanged.connect(self.apply_settings)
        self.actuators_y_input.textChanged.connect(self.apply_settings)

        # PSF Settings group
        psf_group = QGroupBox("PSF Settings")
        psf_layout = QVBoxLayout()

        # Normalization row (horizontal layout)
        norm_layout = QHBoxLayout()
        norm_label = QLabel("Normalization")
        self.norm_combo = QComboBox()
        self.norm_combo.addItems(["Sum Intensity", "Max Value"])
        self.norm_combo.currentIndexChanged.connect(self.update_plots)
        norm_layout.addWidget(norm_label)
        norm_layout.addWidget(self.norm_combo)
        psf_layout.addLayout(norm_layout)

        # Padding factor
        padding_layout = QHBoxLayout()
        padding_layout.addWidget(QLabel("Padding Factor"))
        self.padding_factor_input = QLineEdit(str(self.padding_factor))
        self.padding_factor_input.textChanged.connect(self.apply_settings)
        padding_layout.addWidget(self.padding_factor_input)
        psf_layout.addLayout(padding_layout)

        self.log_scale_checkbox = QCheckBox("Log Scale PSF")
        self.log_scale_checkbox.setChecked(False)
        self.log_scale_checkbox.stateChanged.connect(self.update_plots)
        psf_layout.addWidget(self.log_scale_checkbox)

        psf_group.setLayout(psf_layout)
        control_layout.addWidget(psf_group)

        # Zernike Coefficients inside a scroll area within the group box
        slider_group = QGroupBox("Zernike Coefficients")
        slider_vlayout = QVBoxLayout()

        self.zernike_controls = []
        for i in range(self.num_zernike):
            h = QHBoxLayout()
            lbl = QLabel(f"Z{i}")
            h.addWidget(lbl)

            s = QSlider(Qt.Horizontal)
            s.setMinimum(-100)
            s.setMaximum(100)
            s.setValue(int(self.coefficients[i]*100))

            spin = QDoubleSpinBox()
            spin.setRange(-1.0, 1.0)
            spin.setSingleStep(0.01)
            spin.setValue(self.coefficients[i])

            s.valueChanged.connect(lambda val, idx=i: self.slider_changed(idx, val))
            spin.valueChanged.connect(lambda val, idx=i: self.spinbox_changed(idx, val))

            h.addWidget(s)
            h.addWidget(spin)
            slider_vlayout.addLayout(h)

            self.zernike_controls.append((s, spin))

        # Reset button
        reset_button = QPushButton("Reset Zernike")
        reset_button.clicked.connect(self.reset_zernike)
        slider_vlayout.addWidget(reset_button)

        slider_group.setLayout(slider_vlayout)

        scroll_area = CustomScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.addWidget(slider_group)
        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_widget)

        control_layout.addWidget(scroll_area)

        control_layout.addStretch()
        main_layout.addLayout(control_layout)

        self.initialize_simulation()
        self.update_plots()

        self.setWindowTitle("Interactive Deformable Mirror Simulation")
        self.resize(1200, 600)
        self.show()

    def build_arrays(self):
        grid_size = self.fixed_grid_size
        x = np.linspace(-1, 1, grid_size)
        y = np.linspace(-1, 1, grid_size)
        self.X, self.Y = np.meshgrid(x, y)
        self.R = np.sqrt(self.X**2 + self.Y**2)
        self.Theta = np.arctan2(self.Y, self.X)
        self.mask = self.R <= 1

    def initialize_simulation(self):
        actuator_positions = generate_actuator_positions(
            self.arrangement_type, 
            self.num_actuators_x, 
            self.num_actuators_y, 
            self.extend_outside
        )
        self.zernike_gen = ZernikeGenerator(
            self.zernike_indices, 
            self.R, 
            self.Theta, 
            self.mask
        )
        self.dm_model = DeformableMirror(
            actuator_positions, 
            self.sigma, 
            self.X, 
            self.Y, 
            self.mask
        )
        self.psf_sim = PSFSimulator(wavelength=self.wavelength)

        A = self.dm_model.influence_matrix
        self.Q, self.R_mat = np.linalg.qr(A)

    def solve_fast(self, b):
        lambda_reg = 1e-6
        I = np.eye(self.R_mat.shape[0])
        x = np.linalg.solve(self.R_mat + lambda_reg*I, self.Q.T @ b)
        return x

    def normalize_psf(self, psf):
        mode = self.norm_combo.currentText()
        if mode == "Max Value":
            max_val = psf.max()
            if max_val > 0:
                psf = psf / max_val
        elif mode == "Sum Intensity":
            s = psf.sum()
            if s > 0:
                psf = psf / s
        return psf

    def update_plots(self):
        wavefront = self.zernike_gen.generate_wavefront(self.coefficients)
        b = wavefront[self.mask]
        fitted_coefficients = self.solve_fast(b)
        dm_shape = self.dm_model.deformable_mirror(fitted_coefficients)
        residual_error = wavefront - dm_shape

        use_log = self.log_scale_checkbox.isChecked()
        norm = LogNorm(vmin=1e-6, vmax=1) if use_log else None

        try:
            padding_factor = int(self.padding_factor_input.text())
        except ValueError:
            padding_factor = self.padding_factor

        crop_size = self.crop_size

        original_psf = self.psf_sim.simulate_psf(
            wavefront, 
            self.mask, 
            grid_size=self.fixed_grid_size, 
            padding_factor=padding_factor, 
            crop_size=crop_size
        )
        dm_psf = self.psf_sim.simulate_psf(
            dm_shape, 
            self.mask, 
            grid_size=self.fixed_grid_size, 
            padding_factor=padding_factor, 
            crop_size=crop_size
        )
        residual_psf = self.psf_sim.simulate_psf(
            residual_error, 
            self.mask, 
            grid_size=self.fixed_grid_size, 
            padding_factor=padding_factor, 
            crop_size=crop_size
        )

        original_psf = self.normalize_psf(original_psf)
        dm_psf = self.normalize_psf(dm_psf)
        residual_psf = self.normalize_psf(residual_psf)

        ax = self.canvas.ax
        for a in ax.flatten():
            a.clear()

        vmin, vmax = -1, 1
        ax[0,0].imshow(
            wavefront, 
            extent=(-1, 1, -1, 1), 
            origin='lower', 
            cmap='RdBu', 
            vmin=vmin, 
            vmax=vmax
        )
        ax[0,1].imshow(
            dm_shape, 
            extent=(-1, 1, -1, 1), 
            origin='lower', 
            cmap='RdBu', 
            vmin=vmin, 
            vmax=vmax
        )
        ax[0,2].imshow(
            residual_error, 
            extent=(-1, 1, -1, 1), 
            origin='lower', 
            cmap='RdBu', 
            vmin=vmin, 
            vmax=vmax
        )

        act_pos = self.dm_model.actuator_positions
        ax[0,1].plot(
            [xc for (xc, yc) in act_pos],
            [yc for (xc, yc) in act_pos],
            'ko', markersize=2, markerfacecolor='none'
        )

        ax[1,0].imshow(
            original_psf, 
            origin='lower', 
            cmap='viridis', 
            extent=(-0.05, 0.05, -0.05, 0.05), 
            norm=norm
        )
        ax[1,1].imshow(
            dm_psf, 
            origin='lower', 
            cmap='viridis', 
            extent=(-0.05, 0.05, -0.05, 0.05), 
            norm=norm
        )
        ax[1,2].imshow(
            residual_psf, 
            origin='lower', 
            cmap='viridis', 
            extent=(-0.05, 0.05, -0.05, 0.05), 
            norm=norm
        )

        ax[0,0].set_title("Original Wavefront")
        ax[0,1].set_title("DM Approximation")
        ax[0,2].set_title("Residual Error")
        ax[1,0].set_title("Original PSF")
        ax[1,1].set_title("DM PSF")
        ax[1,2].set_title("Residual PSF")

        for a_ in ax.flatten():
            a_.axis('off')

        self.canvas.draw()

    def apply_settings(self):
        try:
            self.wavelength = float(self.wavelength_input.text())
            self.sigma = float(self.sigma_input.text())
            self.arrangement_type = self.arrangement_input.currentText()
            self.num_actuators_x = int(self.actuators_x_input.text())
            self.num_actuators_y = int(self.actuators_y_input.text())
            self.padding_factor = int(self.padding_factor_input.text())
        except ValueError:
            # If invalid input, do nothing and return
            return

        self.build_arrays()
        self.initialize_simulation()
        self.update_plots()

    def update_sliders(self):
        # Update coefficients from sliders/spin boxes
        for i, (s, spin) in enumerate(self.zernike_controls):
            self.coefficients[i] = spin.value()
        self.update_plots()

    def slider_changed(self, idx, val):
        # Slider is -100 to 100, spin is -1.0 to 1.0
        new_val = val / 100.0
        spin = self.zernike_controls[idx][1]
        if abs(spin.value() - new_val) > 1e-9:
            spin.blockSignals(True)
            spin.setValue(new_val)
            spin.blockSignals(False)
        self.coefficients[idx] = new_val
        self.update_plots()

    def spinbox_changed(self, idx, val):
        # Spin box changed: update slider
        slider = self.zernike_controls[idx][0]
        s_val = int(val * 100)
        if slider.value() != s_val:
            slider.blockSignals(True)
            slider.setValue(s_val)
            slider.blockSignals(False)
        self.coefficients[idx] = val
        self.update_plots()

    def reset_zernike(self):
        for i, (s, spin) in enumerate(self.zernike_controls):
            s.blockSignals(True)
            s.setValue(0)
            s.blockSignals(False)

            spin.blockSignals(True)
            spin.setValue(0.0)
            spin.blockSignals(False)

            self.coefficients[i] = 0.0
        self.update_plots()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
