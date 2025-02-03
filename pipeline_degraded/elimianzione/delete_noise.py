from pipeline_degraded.metric_utils import detect_noises

def is_image_noisy(image):
    blurry_metrics = detect_noises(image)
    noise_image = False

    if 150 <= blurry_metrics["laplacian_variance"] <= 5000:
        pass
    elif blurry_metrics["laplacian_variance"] > 5000:
        if not noise_image:
            noise_image = True
    elif blurry_metrics["laplacian_variance"] < 150:
        if not noise_image:
            noise_image = True

    if 200 <= blurry_metrics["gradient_mean"] <= 1250:
        pass
    elif blurry_metrics["gradient_mean"] > 1250:
        if not noise_image:
            noise_image = True
    elif blurry_metrics["gradient_mean"] < 200:
        if not noise_image:
            noise_image = True

    if blurry_metrics["gdf_entropy"] > 4.5:
        if not noise_image:
            noise_image = True

    if blurry_metrics["gradient_std"] < 450:
        if not noise_image:
            noise_image = True
    return noise_image