import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
from numpy.fft import fft2, ifftshift, fftshift


class ZernikeGenerator:
	"""
	Class to handle Zernike polynomial generation and wavefront construction.
	"""

	def __init__(self, zernike_indices):
		"""
		Initialize with a set of (n, m) Zernike mode indices.

		Parameters:
			zernike_indices (list of tuples): Each tuple is (n, m).
		"""
		self.zernike_indices = zernike_indices

	def zernike_radial(self, n, m, rho):
		"""
		Compute the radial part of the Zernike polynomial.
		"""
		R = np.zeros_like(rho)
		for k in range((n - abs(m)) // 2 + 1):
			c = ((-1)**k * math.factorial(n - k) /
			     (math.factorial(k) *
			      math.factorial((n + abs(m)) // 2 - k) *
			      math.factorial((n - abs(m)) // 2 - k)))
			R += c * rho**(n - 2 * k)
		return R

	def zernike(self, n, m, rho, theta):
		"""
		Compute the full Zernike polynomial Z_n^m(rho, theta).
		"""
		R = self.zernike_radial(n, abs(m), rho)
		if m > 0:
			return R * np.cos(m * theta)
		elif m < 0:
			return R * np.sin(abs(m) * theta)
		else:
			return R

	def generate_wavefront(self, coefficients, rho, theta, mask):
		"""
		Generate a real wavefront from given Zernike coefficients.
		"""
		wavefront = np.zeros_like(rho)
		for coef, (n, m) in zip(coefficients, self.zernike_indices):
			if abs(coef) > 1e-8:
				wavefront += coef * self.zernike(n, m, rho, theta)
		return wavefront * mask


class DeformableMirror:
	"""
	Class to handle deformable mirror (DM) influence and shape approximation.
	"""

	def __init__(self, actuator_positions, sigma):
		"""
		Initialize with actuator positions and influence function parameter.

		Parameters:
			actuator_positions (list): List of (x_center, y_center) tuples.
			sigma (float): Gaussian standard deviation for influence.
		"""
		self.actuator_positions = actuator_positions
		self.sigma = sigma

	def actuator_influence(self, x, y, xc, yc):
		"""
		Compute the influence function of a single DM actuator.
		"""
		return np.exp(-((x - xc)**2 + (y - yc)**2) / (2 * self.sigma**2))

	def deformable_mirror(self, coefficients, X, Y, mask):
		"""
		Compute the DM shape from actuator coefficients.
		"""
		dm_shape = np.zeros_like(X)
		for coeff, (xc, yc) in zip(coefficients, self.actuator_positions):
			dm_shape += coeff * self.actuator_influence(X, Y, xc, yc)
		return dm_shape * mask


class PSFSimulator:
	"""
	Class to handle PSF simulation from wavefronts using Fourier optics.
	"""

	def __init__(self, wavelength):
		"""
		Initialize the PSF Simulator.

		Parameters:
			wavelength (float): Wavelength value used for phase calculation.
		"""
		self.wavelength = wavelength

	def crop_center(self, psf, crop_size=128):
		"""
		Crop the center of the PSF to a specified size.
		"""
		center = psf.shape[0] // 2
		half_crop = crop_size // 2
		return psf[center - half_crop:center + half_crop,
		           center - half_crop:center + half_crop]

	def simulate_psf(self, wavefront, mask, grid_size=200, padding_factor=4):
		"""
		Simulate the PSF from a wavefront using Fourier optics.
		"""
		real_part = np.cos((2 * np.pi / self.wavelength) * wavefront)
		img_part = np.sin((2 * np.pi / self.wavelength) * wavefront)
		complex_wavefront = (real_part + 1j * img_part) * mask

		padded_size = grid_size * padding_factor
		padded_wavefront = np.zeros((padded_size, padded_size), dtype=np.complex128)

		offset = (padded_size - grid_size) // 2
		padded_wavefront[offset:offset + grid_size, offset:offset + grid_size] = complex_wavefront

		fft_wavefront = fftshift(fft2(ifftshift(padded_wavefront)))
		psf = np.abs(fft_wavefront)**2
		psf /= psf.sum()  # Normalize

		return self.crop_center(psf, crop_size=128)


def interpolate_coefficients(t, num_frames, start, end):
	"""
	Interpolate coefficients between start and end using a cosine interpolation.
	"""
	return start + (end - start) * 0.5 * (1 - np.cos(2 * np.pi * t / num_frames))


def generate_actuator_positions(arrangement_type, num_actuators_x, num_actuators_y, extend_outside):
	"""
	Generate actuator positions based on the chosen arrangement type and whether to extend outside the wavefront.
	
	Parameters:
		arrangement_type (str): "grid" or "circular"
		num_actuators_x (int): Number of actuators along x-direction
		num_actuators_y (int): Number of actuators along y-direction
		extend_outside (bool): If True, actuators extend beyond the -1 to 1 range.
	
	Returns:
		list of (float, float): Actuator positions
	"""
	if arrangement_type == "grid":
		# Decide boundaries for the grid
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
		# Example circular arrangement
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
		raise ValueError("Unknown arrangement_type. Choose 'grid' or 'circular'.")
	return actuator_positions


if __name__ == "__main__":
	# Define parameters 
	wavelength = 0.8    # Wavelength in micrometers
	sigma = 0.15        # Influence of actuators
	arrangement_type = "grid"   # "grid" or "circular"
	extend_outside = False       # If True, actuators extend outside -1 to 1 range
	num_actuators_x = 7
	num_actuators_y = 7
	num_actuators_total = num_actuators_x * num_actuators_y

	# Setup grid and mask
	x = np.linspace(-1, 1, 200)
	y = np.linspace(-1, 1, 200)
	X, Y = np.meshgrid(x, y)
	R = np.sqrt(X**2 + Y**2)
	Theta = np.arctan2(Y, X)
	mask = R <= 1

	# Create actuator positions based on chosen arrangement and extension
	actuator_positions = generate_actuator_positions(arrangement_type, num_actuators_x, num_actuators_y, extend_outside)

	# Zernike indices
	zernike_indices = [(n, m) for n in range(5) for m in range(-n, n + 1, 2)]
	num_zernike = len(zernike_indices[:32])

	# Initialize classes
	zernike_gen = ZernikeGenerator(zernike_indices[:32])
	dm_model = DeformableMirror(actuator_positions, sigma=sigma)
	psf_sim = PSFSimulator(wavelength=wavelength)

	# Generate smooth dynamic coefficients
	num_frames = 100
	initial_coefficients = np.random.uniform(-0.1, 0.1, num_zernike)
	target_coefficients = np.random.uniform(-1, 1, num_zernike)

	dynamic_coefficients = np.array([
		interpolate_coefficients(t, num_frames, initial_coefficients, target_coefficients)
		for t in range(num_frames)
	])

	# Add dynamic tilt
	tilt_x_strength = 1
	tilt_y_strength = 1
	for frame in range(num_frames):
		dynamic_coefficients[frame, 1] += tilt_x_strength * np.cos(
			2 * np.pi * frame / num_frames)
		dynamic_coefficients[frame, 2] += tilt_y_strength * np.sin(
			2 * np.pi * frame / num_frames)

	# Set up plots
	fig, ax = plt.subplots(2, 3, figsize=(18, 12))
	vmin, vmax = -2, 2
	initial_wavefront = np.full_like(R, np.nan)
	initial_psf = np.zeros((128, 128))

	original_plot = ax[0, 0].imshow(
		initial_wavefront, extent=(-1, 1, -1, 1),
		origin='lower', cmap='RdBu', vmin=vmin, vmax=vmax
	)
	dm_plot = ax[0, 1].imshow(
		initial_wavefront, extent=(-1, 1, -1, 1),
		origin='lower', cmap='RdBu', vmin=vmin, vmax=vmax
	)
	residual_plot = ax[0, 2].imshow(
		initial_wavefront, extent=(-1, 1, -1, 1),
		origin='lower', cmap='RdBu', vmin=vmin, vmax=vmax
	)

	# Mark actuator positions on the DM plot
	ax[0, 1].plot(
		[xc for (xc, yc) in actuator_positions],
		[yc for (xc, yc) in actuator_positions],
		'ko', markersize=4, markerfacecolor='none'
	)

	original_psf_plot = ax[1, 0].imshow(
		initial_psf, origin='lower', cmap='viridis',
		extent=(-0.05, 0.05, -0.05, 0.05),
		norm=LogNorm(vmin=1e-6, vmax=1)
	)
	dm_psf_plot = ax[1, 1].imshow(
		initial_psf, origin='lower', cmap='viridis',
		extent=(-0.05, 0.05, -0.05, 0.05),
		norm=LogNorm(vmin=1e-6, vmax=1)
	)
	residual_psf_plot = ax[1, 2].imshow(
		initial_psf, origin='lower', cmap='viridis',
		extent=(-0.05, 0.05, -0.05, 0.05),
		norm=LogNorm(vmin=1e-6, vmax=1)
	)

	ax[0, 0].set_title("Original Wavefront")
	ax[0, 1].set_title(f"DM Approximation ({arrangement_type.capitalize()} Actuators)")
	ax[0, 2].set_title("Residual Error")
	ax[1, 0].set_title("Original Wavefront PSF")
	ax[1, 1].set_title("DM PSF")
	ax[1, 2].set_title("Residual PSF")
	for a in ax.flatten():
		a.axis('off')

	def update(frame):
		"""
		Update function for the animation.
		"""
		current_coefficients = dynamic_coefficients[frame]
		original_wavefront = zernike_gen.generate_wavefront(current_coefficients, R, Theta, mask)

		# Build matrix for least squares fitting
		A = np.zeros((np.sum(mask), num_actuators_total))
		for j, (xc, yc) in enumerate(actuator_positions):
			A[:, j] = dm_model.actuator_influence(X[mask], Y[mask], xc, yc)

		fitted_coefficients, _, _, _ = np.linalg.lstsq(A, original_wavefront[mask], rcond=None)
		dm_shape = dm_model.deformable_mirror(fitted_coefficients, X, Y, mask)
		residual_error = original_wavefront - dm_shape

		original_psf = psf_sim.simulate_psf(original_wavefront, mask, grid_size=200, padding_factor=4)
		dm_psf = psf_sim.simulate_psf(dm_shape, mask, grid_size=200, padding_factor=4)
		residual_psf = psf_sim.simulate_psf(residual_error, mask, grid_size=200, padding_factor=4)

		original_plot.set_array(original_wavefront)
		dm_plot.set_array(dm_shape)
		residual_plot.set_array(residual_error)
		original_psf_plot.set_array(original_psf)
		dm_psf_plot.set_array(dm_psf)
		residual_psf_plot.set_array(residual_psf)

		return (
			original_plot, dm_plot, residual_plot,
			original_psf_plot, dm_psf_plot, residual_psf_plot
		)

	ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False)
	#ani.save('animation.gif', fps=5) 

	plt.show()
