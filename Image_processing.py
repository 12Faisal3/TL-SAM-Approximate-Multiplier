import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os

# =========================================================
# TLAM-8 Approximation
# =========================================================
def tlam8(x, y, shift):
    x_t = x >> shift
    y_t = y >> shift
    return (x_t * y_t) << (2 * shift)


# =========================================================
# 16-bit Segmented Multiplier
# =========================================================
def mult16_tlam(A, B, mode="HA"):

    AH, AL = (A >> 8) & 0xFF, A & 0xFF
    BH, BL = (B >> 8) & 0xFF, B & 0xFF

    def exact(x, y):
        return x * y

    if mode == "Exact":
        return A * B

    if mode == "HA":
        s = 1
        PHH = exact(AH, BH)
        PHL = exact(AH, BL)
        PLH = exact(AL, BH)
        PLL = tlam8(AL, BL, s)

    elif mode == "MA":
        s = 2
        PHH = exact(AH, BH)
        PHL = exact(AH, BL)
        PLH = tlam8(AL, BH, s)
        PLL = tlam8(AL, BL, s)

    elif mode == "LA":
        s = 2
        PHH = exact(AH, BH)
        PHL = tlam8(AH, BL, s)
        PLH = tlam8(AL, BH, s)
        PLL = tlam8(AL, BL, s+1)

    elif mode == "FA":
        s = 4
        PHH = tlam8(AH, BH, s)
        PHL = tlam8(AH, BL, s)
        PLH = tlam8(AL, BH, s)
        PLL = tlam8(AL, BL, s)

    return (PHH << 16) + ((PHL + PLH) << 8) + PLL


# =========================================================
# Convolution (Gaussian only)
# =========================================================
def custom_conv(img, kernel, mode):

    h, w = img.shape
    pad = 1

    padded = np.pad(img, pad, mode='edge')
    out = np.zeros_like(img, dtype=np.float32)

    SCALE = 256

    for i in range(h):
        for j in range(w):

            acc = 0

            for m in range(3):
                for n in range(3):
                    pixel = int(padded[i+m, j+n])
                    weight = int(kernel[m, n] * SCALE)

                    acc += mult16_tlam(pixel, weight, mode)

            out[i, j] = acc / SCALE

    return np.clip(out, 0, 255).astype(np.uint8)


# =========================================================
# Gaussian Kernel
# =========================================================
gaussian = np.array([[1,2,1],
                     [2,4,2],
                     [1,2,1]]) / 16


# =========================================================
# Load Image
# =========================================================
img = cv2.imread('/content/cameraman.png', 0)

if img is None:
    raise ValueError("Image not found. Check file path.")


# =========================================================
# Process + Save PDF
# =========================================================
def process_and_save():

    print("Saving to:", os.getcwd())

    exact_out = custom_conv(img, gaussian, "Exact")
    modes = ["HA", "MA", "LA", "FA"]

    fig = plt.figure(figsize=(14,4))
    plt.suptitle("Gaussian Blur", fontsize=14)

    # Original
    plt.subplot(1,6,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # Exact
    plt.subplot(1,6,2)
    plt.imshow(exact_out, cmap='gray')
    plt.title("Exact")
    plt.axis('off')

    # Approx modes
    for i, m in enumerate(modes):
        out = custom_conv(img, gaussian, m)

        p = psnr(exact_out, out)
        s = ssim(exact_out, out)

        plt.subplot(1,6,i+3)
        plt.imshow(out, cmap='gray')
        plt.title(f"{m}\nPSNR={p:.2f}\nSSIM={s:.3f}")
        plt.axis('off')

    plt.tight_layout()

    # SAVE FIGURE
    filename = "gaussian_result.pdf"
    fig.savefig(filename, dpi=300, bbox_inches='tight')

    plt.close(fig)

    print(f"Saved successfully: {filename}")


# =========================================================
# Run
# =========================================================
process_and_save()
