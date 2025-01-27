import cv2
import numpy as np
from scipy.signal import fftconvolve, convolve2d as conv2
from skimage import color, data, restoration
from PIL import Image


def generate_gaussian_psf(shape, sigma):
    """
    Generates a Gaussian PSF (Point Spread Function).

    Args:
        shape (tuple): Shape of the PSF (e.g., (5, 5)).
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        numpy.ndarray: Gaussian PSF array normalized to sum to 1.
    """
    ax = np.arange(-shape[0] // 2 + 1., shape[0] // 2 + 1.)
    ay = np.arange(-shape[1] // 2 + 1., shape[1] // 2 + 1.)
    xx, yy = np.meshgrid(ax, ay)
    psf = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    psf /= psf.sum()
    return psf

def pad_psf(psf, target_shape):
    """
    Pads the PSF (Point Spread Function) to match the target shape of the image.

    Args:
        psf (numpy.ndarray): Initial PSF array (e.g., 5x5).
        target_shape (tuple): Shape of the target image (e.g., 256x455).

    Returns:
        numpy.ndarray: Padded PSF with the same shape as the target image.
    """
    padded_psf = np.zeros(target_shape, dtype=np.float32)
    psf_shape = psf.shape
    center = [s // 2 for s in psf_shape]

    # Place the PSF in the center of the padded array
    padded_psf[:psf_shape[0], :psf_shape[1]] = psf
    padded_psf = np.roll(padded_psf, -center[0], axis=0)
    padded_psf = np.roll(padded_psf, -center[1], axis=1)

    return padded_psf / padded_psf.sum()  # Normalize PSF to maintain proper distribution


def richardson_lucy_blind(image, psf, num_iter=50):
    """
    Perform blind deconvolution using the accelerated, damped Richardson-Lucy algorithm.

    Parameters:
        image (numpy.ndarray): Input blurred image.
        psf (numpy.ndarray): Initial guess for the point-spread function (PSF).
        num_iter (int): Number of iterations.

    Returns:
        deconvolved (numpy.ndarray): Restored image.
        psf (numpy.ndarray): Estimated point-spread function.
    """
    psf = pad_psf(psf, image.shape)
    # Initialize restored image with a uniform value
    deconvolved = np.full(image.shape, 0.1, dtype=np.float32)

    for i in range(num_iter):
        # Convolve the current estimate of the image with the PSF
        estimated_blur = fftconvolve(deconvolved, psf, mode='same')

        # Avoid division by zero
        estimated_blur[estimated_blur == 0] = 1e-10

        # Compute the ratio of the blurred image to the estimate
        relative_blur = image / estimated_blur

        # Update the restored image
        psf_mirror = np.flip(psf)
        deconvolved *= fftconvolve(relative_blur, psf_mirror, mode='same')

        # Normalize restored image to prevent overflow
        deconvolved /= deconvolved.sum()

        # Update the PSF using the flipped image
        deconvolved_mirror = np.flip(deconvolved)
        psf_update = fftconvolve(relative_blur, deconvolved_mirror, mode='same')
        # Normalize the PSF to ensure it's a valid probability distribution
        psf *= psf_update
        psf /= psf.sum()

    return deconvolved, psf


def blurriness_1(image: Image.Image):
    img = image.convert("L")
    image = np.array(img, dtype=np.float32)
    psf_initial = generate_gaussian_psf((5, 5), sigma=2)
    deconvolved_image, estimated_psf = richardson_lucy_blind(image, psf_initial, num_iter=50)
    deconvolved_pil = Image.fromarray(np.clip(deconvolved_image, 0, 255).astype(np.uint8))
    return deconvolved_pil


def blurriness_2(image: Image.Image):
    img_np =  np.array(image)
    gray_image = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    psf = cv2.getGaussianKernel(5, sigma=1)
    psf = psf @ psf.T
    deconvolved_RL = restoration.richardson_lucy(gray_image, psf, num_iter=30)
    # deconvolved_RL = (deconvolved_RL - deconvolved_RL.min()) / (deconvolved_RL.max() - deconvolved_RL.min()) * 255
    # deconvolved_image = Image.fromarray(deconvolved_RL.astype(np.uint8), mode='L')
    return deconvolved_RL


def apply_median_filter(image, n=3):
    d = n // 2

    # Convert PIL image to NumPy array
    image = np.array(image)
    m1, n1, w = image.shape

    # Create padded images
    m2, n2 = m1 + 2 * d, n1 + 2 * d
    image_n = np.zeros((m2, n2, w), dtype=np.float32)
    image_n1 = np.zeros((m2, n2, w), dtype=np.float32)

    # Pad the original image
    image_n[d:m2 - d, d:n2 - d, :] = image

    # Apply the filter
    mm = d
    for k in range(w):
        t = 0
        for i in range(m1):
            r = 0
            for j in range(n1):
                # Extract the local window
                b = image_n[t:t + n, r:r + n, k]
                s = np.std(b)
                m = np.mean(b)

                # Check condition and calculate median
                if b[d, d] > m + 0.01 * s or b[d, d] < m - 0.01 * s:
                    B = b.flatten()
                    B = np.delete(B, len(B) // 2)
                    st = np.median(B)
                    image_n1[mm + t, mm + r, k] = st
                else:
                    image_n1[mm + t, mm + r, k] = 0

                r += 1
            t += 1

    # Combine the filtered and original images
    for k in range(w):
        for i in range(m1):
            for j in range(n1):
                if image_n1[i + d, j + d, k] != 0:
                    image_n[i + d, j + d, k] = image_n1[i + d, j + d, k]

    # Clip values and convert to uint8
    image_n = np.clip(image_n[d:m2 - d, d:n2 - d, :], 0, 255).astype(np.uint8)

    # Convert back to PIL image
    filtered_image = Image.fromarray(image_n)
    return filtered_image