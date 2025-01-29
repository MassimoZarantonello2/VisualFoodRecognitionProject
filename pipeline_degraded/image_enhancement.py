import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

def denoise_salt_pepper(image: Image.Image, kernel_size: int = 3) -> Image.Image:
    """
    Applies a median filter to the input image to reduce salt-and-pepper noise.

    Args:
        image (Image.Image): Input PIL image.
        kernel_size (int): Size of the kernel (must be an odd integer). Default is 3.

    Returns:
        Image.Image: Denoised PIL image.
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd integer.")

    image_np = np.array(image)

    if len(image_np.shape) == 2:
        denoised_np = cv2.medianBlur(image_np, kernel_size)
    elif len(image_np.shape) == 3:
        channels = cv2.split(image_np)
        denoised_channels = [cv2.medianBlur(ch, kernel_size) for ch in channels]
        denoised_np = cv2.merge(denoised_channels)
    else:
        raise ValueError("Unsupported image format.")

    image_rgb = cv2.cvtColor(denoised_np, cv2.COLOR_BGR2RGB)

    denoised_image = Image.fromarray(image_rgb)

    return denoised_image

def non_local_means(image):
    image = np.array(image)
    denoised_nlmeans = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    nlmeans_rgb = cv2.cvtColor(denoised_nlmeans, cv2.COLOR_BGR2RGB)
    denoised_image = Image.fromarray(nlmeans_rgb)
    return denoised_image

def denoise_bilateral(image):
    image = np.array(image)
    denoised_bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    bilateral_rgb = cv2.cvtColor(denoised_bilateral, cv2.COLOR_BGR2RGB)
    denoised_image = Image.fromarray(bilateral_rgb)
    return denoised_image

def enhance_sharpness(image, factor=2.0):
    enhancer = ImageEnhance.Sharpness(image)
    sharpened_image = enhancer.enhance(factor)
    return sharpened_image

def denoise_bilateral_mask(image):
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    filtered_mask = cv2.bilateralFilter(mask, 9, 75, 75)
    denoised_image = image.copy()
    denoised_image[mask > 0] = cv2.cvtColor(filtered_mask, cv2.COLOR_GRAY2BGR)[mask > 0]
    final_image = Image.fromarray(cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB))
    return final_image

def adaptive_median_filter(image_pil, S_max=7):
    """
    Apply an adaptive median filter to each channel of an RGB image.

    Args:
        image_pil (PIL.Image): Input image in PIL format (RGB).
        S_max (int): Maximum window size for the adaptive filter.

    Returns:
        PIL.Image: Filtered image in PIL format (RGB).
    """
    image_array = np.array(image_pil)

    if len(image_array.shape) != 3 or image_array.shape[2] != 3:
        raise ValueError("Input image must be a color image in RGB format.")

    filtered_array = np.zeros_like(image_array)

    for channel in range(3):
        channel_data = image_array[:, :, channel]

        padded_channel = np.pad(channel_data, S_max // 2, mode='constant', constant_values=0)
        output_channel = np.copy(channel_data)

        rows, cols = channel_data.shape
        for i in range(rows):
            for j in range(cols):
                S = 3
                while S <= S_max:
                    sub_img = padded_channel[i:i + S, j:j + S]
                    Z_min = np.min(sub_img)
                    Z_max = np.max(sub_img)
                    Z_m = np.median(sub_img)
                    Z_xy = channel_data[i, j]

                    if Z_min < Z_m < Z_max:
                        if Z_min < Z_xy < Z_max:
                            output_channel[i, j] = Z_xy
                        else:
                            output_channel[i, j] = Z_m
                        break
                    else:
                        S += 2
                else:
                    output_channel[i, j] = Z_m

        filtered_array[:, :, channel] = output_channel

    filtered_image_pil = Image.fromarray(filtered_array)

    return filtered_image_pil


def adaptive_gamma_correction(image: Image.Image, sigma_color: float = 0.1, sigma_space: float = 15, saturation_factor: float = 1.5) -> Image.Image:
    """
    Applies adaptive gamma correction with a bilateral filter to the input image.

    Args:
        image (Image.Image): Input PIL image.
        sigma_color (float): Filter sigma in the color space for the bilateral filter.
        sigma_space (float): Filter sigma in the coordinate space for the bilateral filter.
        saturation_factor (float): Factor to adjust saturation in the HSV color space.

    Returns:
        Image.Image: Image after adaptive gamma correction.
    """
    image_np = np.array(image)

    ycbcr_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2YCrCb)

    y_channel = ycbcr_image[:, :, 0] / 255.0

    inverted_mask = 1.0 - y_channel
    bilateral_mask = cv2.bilateralFilter(inverted_mask.astype(np.float32), d=0, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    mean_intensity = np.mean(gray_image)

    if mean_intensity < 128:
        alpha = np.log(mean_intensity / 255.0) / np.log(0.5)
    else:
        alpha = np.log(0.5) / np.log(mean_intensity / 255.0)

    adaptive_gamma = alpha ** ((0.5 - bilateral_mask) / 0.5)
    corrected_y_channel = np.power(y_channel, adaptive_gamma)

    ycbcr_image[:, :, 0] = np.clip(corrected_y_channel * 255.0, 0, 255).astype(np.uint8)

    corrected_image = cv2.cvtColor(ycbcr_image, cv2.COLOR_YCrCb2RGB)

    hsv_image = cv2.cvtColor(corrected_image, cv2.COLOR_RGB2HSV)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255).astype(np.uint8)

    final_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    result_image = Image.fromarray(final_image)

    return result_image

def blind_deconvolution(image, iterations=25, psf_size=(5, 5)):
    """
    Perform Blind Deconvolution using the Richardson-Lucy algorithm.

    Args:
        image (Image.Image): Input PIL image.
        iterations (int): Number of iterations for the deconvolution algorithm.
        psf_size (tuple): Size of the Point Spread Function (PSF).

    Returns:
        Image.Image: Enhanced image after deconvolution.
    """
    img = np.array(image.convert('L'), dtype=np.float32) / 255.0  # Normalize to [0, 1]

    # Initialize a uniform Point Spread Function (PSF)
    psf = np.ones(psf_size, dtype=np.float32)
    psf /= np.sum(psf)

    # Function to perform the Richardson-Lucy deconvolution
    def richardson_lucy(image, psf, iterations):
        estimate = np.full_like(image, 0.5)
        psf_mirror = np.flip(psf)

        for _ in range(iterations):
            # Convolve estimate with PSF
            convolved = cv2.filter2D(estimate, -1, psf, borderType=cv2.BORDER_REPLICATE)
            # Compute relative blur
            relative_blur = image / (convolved + 1e-7)
            # Update estimate
            correction = cv2.filter2D(relative_blur, -1, psf_mirror, borderType=cv2.BORDER_REPLICATE)
            estimate *= correction

        return estimate

    deconvolved = richardson_lucy(img, psf, iterations)
    deconvolved = (deconvolved * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(deconvolved)



def colorize_deconvolved_image(original_image, deconvolved_image):
    """
    Colorize the deconvolved grayscale image using the chrominance channels (Cb, Cr) of the original image.

    Args:
        original_image : Original color image (PIL format).
        deconvolved_image : Grayscale deconvolved image (PIL format).

    Returns:
        Image.Image: Colorized version of the deconvolved image.
    """
    # Convert original image to OpenCV format (RGB)
    original_cv = np.array(original_image, dtype=np.float32) / 255.0  # Normalize to [0, 1]

    ycbcr = cv2.cvtColor(original_cv, cv2.COLOR_RGB2YCrCb)
    _, cb, cr = cv2.split(ycbcr)
    deconvolved_array = np.array(deconvolved_image, dtype=np.float32) / 255.0

    if deconvolved_array.shape != cb.shape:
        deconvolved_array = cv2.resize(deconvolved_array, (cb.shape[1], cb.shape[0]), interpolation=cv2.INTER_CUBIC)

    enhanced_ycbcr = cv2.merge((deconvolved_array, cb, cr))

    enhanced_rgb = cv2.cvtColor(enhanced_ycbcr, cv2.COLOR_YCrCb2RGB)

    enhanced_image = Image.fromarray((enhanced_rgb * 255).clip(0, 255).astype(np.uint8))

    return enhanced_image

def blurriness(image: Image.Image):
    """
    Computes and applies blind deconvolution techniques to reduce blurriness in a
    given image and restores its clarity. The function works on the input image by
    first enhancing its greyscale version using deconvolution and then restoring
    colors to the enhanced greyscale result.

    Args:
        image (Image.Image): The input image to be deblurred.

    Returns:
        Image.Image: An enhanced version of the input image with reduced blurriness.
    """
    enhanced_image_grey = blind_deconvolution(image)
    enhanced_image = colorize_deconvolved_image(image, enhanced_image_grey)
    return enhanced_image


def white_balancingGrayWorld(image: Image.Image):
    """
    Performs white balancing on an input image using the Gray World algorithm. This method
    adjusts the colors of the image based on the assumption that the average color in the
    scene is gray, leading to more natural white balance. The method operates on OpenCV's
    image representation and converts the processed result back to PIL format.

    Args:
        image (Image.Image): The input image in PIL format. This image will be processed
        and balanced using the Gray World algorithm.

    Returns:
        Image.Image: The white-balanced image in PIL format after applying the Gray World
        algorithm.
    """
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    grayworld = cv2.xphoto.createGrayworldWB()
    balanced_image = grayworld.balanceWhite(image_cv)
    balanced_image_pil = Image.fromarray(cv2.cvtColor(balanced_image, cv2.COLOR_BGR2RGB))
    return balanced_image_pil


def max_rgb(image: Image.Image, saturation = False, sfactor=1):
    """
    Adjusts the RGB channel values of an image to maximize their range and enhances the
    saturation of the image.

    The function takes a PIL image, converts it to OpenCV format, and normalizes
    the R, G, and B channels to maximize color detail. Subsequently, it enhances the
    image's saturation in the HSV color space by a specified scaling factor. The
    resulting image is converted back to a PIL image format.

    Parameters:
    image: Image.Image
        A PIL Image object representing the input image that needs enhancement.
    sfactor: float, optional
        A scaling factor for adjusting image saturation in the HSV color
        representation. Default is 1.5.

    Returns:
    Image.Image
        A PIL Image object representing the enhanced image with adjusted RGB
        channels and increased saturation.
    """
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    max_R = np.max(image_cv[:, :, 2]) or 1
    max_G = np.max(image_cv[:, :, 1]) or 1
    max_B = np.max(image_cv[:, :, 0]) or 1

    coeff = [255.0 / max_R, 255.0 / max_G, 255.0 / max_B]
    image_cv[:, :, 2] = (image_cv[:, :, 2] * coeff[0]).clip(0, 255)
    image_cv[:, :, 1] = (image_cv[:, :, 1] * coeff[1]).clip(0, 255)
    image_cv[:, :, 0] = (image_cv[:, :, 0] * coeff[2]).clip(0, 255)

    # Convert to HSV and enhance saturation, if saturation enable
    image_hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV).astype(np.float32)
    if saturation:
        image_hsv[:, :, 1] = (image_hsv[:, :, 1] * sfactor).clip(0, 255)
    image_cv = cv2.cvtColor(image_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Convert back to PIL image
    enhanced_image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

    return enhanced_image_pil

def deblurring(image):
    image = np.array(image)

    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(image, 0 , sharpen_kernel)
    deblurred = cv2.fastNlMeansDenoisingColored(sharpen,None,10,10,7,21)
    denoised_image = Image.fromarray(deblurred)
    return denoised_image



