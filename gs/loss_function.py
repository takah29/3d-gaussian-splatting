import jax.numpy as jnp
from jax.scipy.signal import convolve


def l2_loss(output, target):
    return jnp.mean(jnp.square(output - target))


def l1_loss(output, target):
    return jnp.mean(jnp.abs(output - target))


def gs_loss(output, target, alpha: float = 0.2):
    return (1 - alpha) * l1_loss(output, target) + alpha * dssim_loss(output, target)


def dssim_loss(output, target, window_size=11):
    return (1 - _ssim(output, target, window_size)) / 2.0


def _create_2d_gaussian_kernel(window_size, sigma=1.5):
    half_size = window_size // 2
    kernel_1d = jnp.exp(jnp.arange(-half_size, half_size + 1) ** 2 / (2 * sigma**2))
    kernel = kernel_1d[:, None] @ kernel_1d[None, :]
    kernel = kernel / kernel.sum()
    return jnp.stack([kernel] * 3, axis=2)  # (window_size, window_size, 3)


def _ssim(img1, img2, window_size):
    kernel = _create_2d_gaussian_kernel(window_size)
    mu1 = convolve(img1, kernel, mode="valid")
    mu2 = convolve(img2, kernel, mode="valid")

    mu1_sq = jnp.square(mu1)
    mu2_sq = jnp.square(mu2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = convolve(jnp.square(img1), kernel, mode="valid") - mu1_sq
    sigma2_sq = convolve(jnp.square(img2), kernel, mode="valid") - mu2_sq
    sigma12 = convolve(img1 * img2, kernel, mode="valid") - mu1_mu2

    c1 = 0.01**2
    c2 = 0.03**2

    elementwise_ssim = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    return elementwise_ssim.mean()
