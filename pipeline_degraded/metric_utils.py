import cv2
import numpy as np
from PIL import Image, ImageFilter
from scipy.stats import kurtosis

def detect_noises(image: Image.Image):
    """
        Detects noises in an image by analyzing its sharpness, gradient distribution,
        and the count of extreme pixel values. This function computes several metrics
        to determine noise characteristics based on image properties and statistics.

        Arguments:
            image (PIL.Image.Image): The input image to process. It should be readable
            and convertible to grayscale for analysis.

        Returns:
            dict: A dictionary containing the calculated noise metrics:
                - laplacian_variance (float): Variance of the Laplacian, a measure of
                  sharpness.
                - gradient_mean (float): Mean value of the gradient magnitude
                  computed from the Sobel filter.
                - gradient_std (float): Standard deviation of the gradient magnitude
                  computed from the Sobel filter.
                - gdf_entropy (float): Entropy of the Gradient Distribution Function
                  (GDF), representing the smoothness and variability of gradients in
                  the image.
                - extreme_pixel (float): Percentage of pixels with intensity values
                  near 0 or 255, indicating extreme dark or bright areas.

        Raises:
            ValueError: If the input image is None or invalid.
    """
    if image is None:
        raise ValueError("Could not open or find the image.")

    image = np.array(image.convert('L'))

    # 1. Laplacian variance method
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian_var = laplacian.var()

    # 2. Gradient-based sharpness calculation
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    gradient_mean = np.mean(gradient_magnitude)
    gradient_std = np.std(gradient_magnitude)

    # 3. Gradient Distribution Function (GDF)
    hist, _ = np.histogram(gradient_magnitude, bins=50, range=(0, np.max(gradient_magnitude)))
    gdf = hist / np.sum(hist)  # Normalize to create a distribution function
    gdf_entropy = -np.sum(gdf * np.log2(gdf + 1e-10))  # Entropy of GDF

    # 4. calculate the percentage of pixels in the image that have intensity values
    #     very close to 0 (black) or 255 (white)
    extreme_pixels = calculate_extreme_pixel(image)
    # 5. Kurtosis and MAD
    pixel_values = image.flatten()
    mean_intensity = np.mean(pixel_values)
    mad = np.mean(np.abs(pixel_values - mean_intensity))

    kurt = kurtosis(pixel_values, fisher=False)
    # 6. Variance of pixel intensities
    variance = np.var(pixel_values)

    # 7. Signal-to-Noise Ratio (SNR)
    signal_power = mean_intensity ** 2
    snr = 10 * np.log10(signal_power / (variance + 1e-10))

    return {
        "laplacian_variance": laplacian_var,
        "gradient_mean": gradient_mean,
        "gradient_std": gradient_std,
        "gdf_entropy": gdf_entropy,
        "extreme_pixel": extreme_pixels,
        "mad": mad,
        "kurt": kurt,
        "variance": variance,
        "snr": snr,
    }


def calculate_extreme_pixel(image_array, lower_threshold=30, upper_threshold=225):
    """
    Calculates the percentage of extreme pixels in a given image array.

    This function determines the proportion of pixels in an input image array
    that are considered to be either very close to black (dark colors) or very
    close to white (light colors) based on given thresholds. It does so by
    measuring the count of pixels falling below the lower threshold and above
    the upper threshold, and calculates their percentage relative to the total
    number of pixels in the image.

    Args:
        image_array: A NumPy array representing the image.
        lower_threshold (default=30): An integer for the lower threshold to
            categorize pixels as close to black.
        upper_threshold (default=225): An integer for the upper threshold to
            categorize pixels as close to white.

    Returns:
        float: The percentage of pixels in the image that are considered extreme
            (close to black or white).
    """

    close_to_black = image_array < lower_threshold
    close_to_white = image_array > upper_threshold

    black_pixel_count = np.sum(close_to_black)
    white_pixel_count = np.sum(close_to_white)
    total_pixels = image_array.size
    extreme_pixel_percentage = ((black_pixel_count + white_pixel_count) / total_pixels)

    return extreme_pixel_percentage