# TL-SAM: Truncated Logarithmic Segmentation-Based Approximate Multiplier 
# *(Remaining code will be available after the acceptance of the work)*

This repository contains the reference implementation of TL-SAM, a configurable truncated logarithmic approximate multiplier architecture designed for energy-efficient Edge-AI and error-resilient computing applications.

## Features

- 8-bit truncated logarithmic approximate multiplier (TLAM-8)
- Segmented 16-bit configurable architecture
- Four approximation modes:
  - HA (High Accuracy)
  - MA (Medium Accuracy)
  - LA (Low Accuracy)
  - FA (Fully Approximate)
- FPGA and ASIC evaluation support
- Python-based error metric evaluation
- Image processing validation
- LeNet-5 inference evaluation

---

## Repository Structure

| Directory | Description |
|---|---|
| RTL | Verilog RTL source files |
| Python | Error metric and validation scripts |
| Images | Input and output image samples |
| Results | Experimental evaluation results |
| Docs | Architecture diagrams and algorithms |

---

## Supported Error Metrics

- NMED
- MRED
- PSNR
- SSIM

---

## FPGA/ASIC Flow

The published repository contains functional RTL for reproducibility purposes. Proprietary synthesis libraries and technology-specific implementation scripts are not included.

---

## Citation

If you use this work in your research, please cite:

```bibtex
@article{TLAM2026,
  title={TL-SAM: Truncated Logarithmic Segmentation-Based Approximate Multiplier for Edge-AI Applications},
  author={Your Name},
  journal={IEEE Embedded Systems Letters},
  year={2026}
}
