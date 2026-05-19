import numpy as np

# =========================================================
# TLAM-8 Approximate Multiplier
# =========================================================
def tlam8(x, y, shift):

    x_t = x >> shift
    y_t = y >> shift

    return (x_t * y_t) << (2 * shift)


# =========================================================
# 16-bit Segmented TL-SAM Multiplier
# =========================================================
def mult16_tlam(A, B, mode="HA"):

    AH, AL = (A >> 8) & 0xFF, A & 0xFF
    BH, BL = (B >> 8) & 0xFF, B & 0xFF

    def exact(x, y):
        return x * y

    if mode == "Exact":
        return A * B

    # -----------------------------------------------------
    # HA Mode
    # -----------------------------------------------------
    if mode == "HA":

        s = 1

        PHH = exact(AH, BH)
        PHL = exact(AH, BL)
        PLH = exact(AL, BH)
        PLL = tlam8(AL, BL, s)

    # -----------------------------------------------------
    # MA Mode
    # -----------------------------------------------------
    elif mode == "MA":

        s = 2

        PHH = exact(AH, BH)
        PHL = exact(AH, BL)
        PLH = tlam8(AL, BH, s)
        PLL = tlam8(AL, BL, s)

    # -----------------------------------------------------
    # LA Mode
    # -----------------------------------------------------
    elif mode == "LA":

        s = 2

        PHH = exact(AH, BH)
        PHL = tlam8(AH, BL, s)
        PLH = tlam8(AL, BH, s)
        PLL = tlam8(AL, BL, s + 1)

    # -----------------------------------------------------
    # FA Mode
    # -----------------------------------------------------
    elif mode == "FA":

        s = 3

        PHH = tlam8(AH, BH, s)
        PHL = tlam8(AH, BL, s)
        PLH = tlam8(AL, BH, s)
        PLL = tlam8(AL, BL, s)

    return (PHH << 16) + ((PHL + PLH) << 8) + PLL


# =========================================================
# Error Metric Calculation
# =========================================================
def calculate_error_metrics(mode):

    total_error_distance = 0
    total_relative_error = 0

    total_cases = 0

    max_output = (2**16 - 1) * (2**16 - 1)

    # -----------------------------------------------------
    # Exhaustive Evaluation
    # -----------------------------------------------------
    for A in range(65536):

        if A % 2000 == 0:
            print(f"Processing A = {A}")

        for B in range(65536):

            exact = A * B

            approx = mult16_tlam(A, B, mode)

            error_distance = abs(exact - approx)

            total_error_distance += error_distance

            if exact != 0:
                total_relative_error += error_distance / exact

            total_cases += 1

    # -----------------------------------------------------
    # NMED
    # -----------------------------------------------------
    NMED = total_error_distance / (total_cases * max_output)

    # -----------------------------------------------------
    # MRED
    # -----------------------------------------------------
    MRED = total_relative_error / total_cases

    return NMED, MRED


# =========================================================
# Run for All Modes
# =========================================================
modes = ["HA", "MA", "LA", "FA"]

print("\n==========================================")
print("TL-SAM Error Metric Evaluation")
print("==========================================\n")

for mode in modes:

    print(f"\nEvaluating Mode: {mode}")

    nmed, mred = calculate_error_metrics(mode)

    print(f"\nMode: {mode}")

    print(f"NMED = {nmed:.6e}")

    print(f"MRED = {mred:.6e}")

    print("\n--------------------------------------")
