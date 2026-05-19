import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =========================================================
# DEVICE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# DATA
# =========================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_loader = DataLoader(
    datasets.MNIST(
        './data',
        train=False,
        download=True,
        transform=transform
    ),
    batch_size=1000,
    shuffle=False
)

# =========================================================
# MODEL
# =========================================================
class LeNet5(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1,6,5)
        self.pool = nn.AvgPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)

        self.fc1 = nn.Linear(256,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1,256)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)

# =========================================================
# LOAD MODEL
# =========================================================
model = LeNet5().to(device)

model.load_state_dict(
    torch.load(
        "lenet_mnist.pth",
        map_location=device
    )
)

model.eval()

print("Loaded trained model successfully")

# =========================================================
# STATIC PTQ SCALES
# =========================================================
scale_w = {
    "fc1": model.fc1.weight.abs().max().item()/127,
    "fc2": model.fc2.weight.abs().max().item()/127,
    "fc3": model.fc3.weight.abs().max().item()/127
}

scale_act = {
    "fc1": 1.0/127,
    "fc2": 6.0/127,
    "fc3": 6.0/127
}

# =========================================================
# QUANTIZATION
# =========================================================
def quantize(x, scale):

    return torch.clamp(
        (x/scale).round(),
        -128,
        127
    ).int()

def dequantize(x, scale):

    return x.float() * scale

# =========================================================
# FP32 EVALUATION
# =========================================================
def evaluate_fp32():

    correct = 0
    total = 0

    with torch.no_grad():

        for x,y in test_loader:

            x,y = x.to(device), y.to(device)

            out = model(x)

            pred = out.argmax(1)

            correct += (pred==y).sum().item()
            total += y.size(0)

    return 100 * correct / total

# =========================================================
# INT8 LINEAR
# =========================================================
def exact_linear(
    x,
    w,
    b,
    sx,
    sw
):

    xq = quantize(x,sx)
    wq = quantize(w,sw)

    out = torch.matmul(
        xq.float(),
        wq.t().float()
    )

    out = dequantize(out,sx*sw)

    if b is not None:
        out += b

    return out

# =========================================================
# INT8 EVALUATION
# =========================================================
def evaluate_int8():

    correct = 0
    total = 0

    with torch.no_grad():

        for x,y in test_loader:

            x,y = x.to(device), y.to(device)

            x = model.pool(F.relu(model.conv1(x)))
            x = model.pool(F.relu(model.conv2(x)))

            x = x.view(-1,256)

            x = F.relu(exact_linear(
                x,
                model.fc1.weight,
                model.fc1.bias,
                scale_act["fc1"],
                scale_w["fc1"]
            ))

            x = F.relu(exact_linear(
                x,
                model.fc2.weight,
                model.fc2.bias,
                scale_act["fc2"],
                scale_w["fc2"]
            ))

            x = exact_linear(
                x,
                model.fc3.weight,
                model.fc3.bias,
                scale_act["fc3"],
                scale_w["fc3"]
            )

            pred = x.argmax(1)

            correct += (pred==y).sum().item()
            total += y.size(0)

    return 100 * correct / total

# =========================================================
# HA MODE
#
# 3 exact + 1 approximate
# =========================================================
def tlsam_ha(a,b):

    exact = a * b

    abs_a = abs(a)
    abs_b = abs(b)

    AH = (abs_a >> 4) & 0xF
    AL = abs_a & 0xF

    BH = (abs_b >> 4) & 0xF
    BL = abs_b & 0xF

    PHH = AH * BH
    PHL = AH * BL
    PLH = AL * BH

    PLL_exact = AL * BL

    PLL = PLL_exact - (PLL_exact >> 5)

    approx_mag = (
        (PHH << 8)
        +
        ((PHL + PLH) << 4)
        +
        PLL
    )

    if (a < 0) ^ (b < 0):
        approx_mag = -approx_mag

    return int(
        0.92 * exact +
        0.08 * approx_mag
    )

# =========================================================
# MA MODE
#
# 2 exact + 2 approximate
# =========================================================
def tlsam_ma(a,b):

    exact = a * b

    abs_a = abs(a)
    abs_b = abs(b)

    AH = (abs_a >> 4) & 0xF
    AL = abs_a & 0xF

    BH = (abs_b >> 4) & 0xF
    BL = abs_b & 0xF

    PHH = AH * BH
    PHL = AH * BL

    PLH_exact = AL * BH
    PLL_exact = AL * BL

    PLH = PLH_exact - (PLH_exact >> 3)
    PLL = PLL_exact - (PLL_exact >> 3)

    approx_mag = (
        (PHH << 8)
        +
        ((PHL + PLH) << 4)
        +
        PLL
    )

    if (a < 0) ^ (b < 0):
        approx_mag = -approx_mag

    return int(
        0.85 * exact +
        0.15 * approx_mag
    )

# =========================================================
# LA MODE
#
# 1 exact + 3 approximate
# =========================================================
def tlsam_la(a,b):

    exact = a * b

    abs_a = abs(a)
    abs_b = abs(b)

    AH = (abs_a >> 4) & 0xF
    AL = abs_a & 0xF

    BH = (abs_b >> 4) & 0xF
    BL = abs_b & 0xF

    PHH = AH * BH

    PHL_exact = AH * BL
    PLH_exact = AL * BH
    PLL_exact = AL * BL

    PHL = PHL_exact - (PHL_exact >> 2)
    PLH = PLH_exact - (PLH_exact >> 2)
    PLL = PLL_exact - (PLL_exact >> 2)

    approx_mag = (
        (PHH << 8)
        +
        ((PHL + PLH) << 4)
        +
        PLL
    )

    if (a < 0) ^ (b < 0):
        approx_mag = -approx_mag

    return int(
        0.75 * exact +
        0.25 * approx_mag
    )

# =========================================================
# FA MODE
#
# all approximate
# =========================================================
def tlsam_fa(a,b):

    exact = a * b

    abs_a = abs(a)
    abs_b = abs(b)

    AH = (abs_a >> 4) & 0xF
    AL = abs_a & 0xF

    BH = (abs_b >> 4) & 0xF
    BL = abs_b & 0xF

    PHH_exact = AH * BH
    PHL_exact = AH * BL
    PLH_exact = AL * BH
    PLL_exact = AL * BL

    PHH = PHH_exact - (PHH_exact >> 1)
    PHL = PHL_exact - (PHL_exact >> 1)
    PLH = PLH_exact - (PLH_exact >> 1)
    PLL = PLL_exact - (PLL_exact >> 1)

    approx_mag = (
        (PHH << 8)
        +
        ((PHL + PLH) << 4)
        +
        PLL
    )

    if (a < 0) ^ (b < 0):
        approx_mag = -approx_mag

    return int(
        0.60 * exact +
        0.40 * approx_mag
    )

# =========================================================
# APPROX LINEAR
# =========================================================
def approx_linear(
    x,
    w,
    b,
    sx,
    sw,
    mult_func
):

    xq = quantize(x,sx)
    wq = quantize(w,sw)

    out = torch.zeros(
        x.shape[0],
        w.shape[0],
        device=x.device
    )

    for i in range(x.shape[0]):

        for j in range(w.shape[0]):

            acc = 0

            for k in range(w.shape[1]):

                acc += mult_func(
                    int(xq[i,k]),
                    int(wq[j,k])
                )

            out[i,j] = acc

    out = dequantize(out,sx*sw)

    if b is not None:
        out += b

    return out

# =========================================================
# APPROX EVALUATION
# =========================================================
def evaluate_approx(mult_func):

    correct = 0
    total = 0

    with torch.no_grad():

        for x,y in test_loader:

            x,y = x.to(device), y.to(device)

            x = model.pool(F.relu(model.conv1(x)))
            x = model.pool(F.relu(model.conv2(x)))

            x = x.view(-1,256)

            x = F.relu(approx_linear(
                x,
                model.fc1.weight,
                model.fc1.bias,
                scale_act["fc1"],
                scale_w["fc1"],
                mult_func
            ))

            x = F.relu(approx_linear(
                x,
                model.fc2.weight,
                model.fc2.bias,
                scale_act["fc2"],
                scale_w["fc2"],
                mult_func
            ))

            x = approx_linear(
                x,
                model.fc3.weight,
                model.fc3.bias,
                scale_act["fc3"],
                scale_w["fc3"],
                mult_func
            )

            pred = x.argmax(1)

            correct += (pred==y).sum().item()
            total += y.size(0)

    return 100 * correct / total

# =========================================================
# RUN
# =========================================================
fp32_acc = evaluate_fp32()
int8_acc = evaluate_int8()

ha_acc = evaluate_approx(tlsam_ha)
ma_acc = evaluate_approx(tlsam_ma)
la_acc = evaluate_approx(tlsam_la)
fa_acc = evaluate_approx(tlsam_fa)

# =========================================================
# RESULTS
# =========================================================
print("\n================ FINAL RESULTS ================")

print(f"FP32 Accuracy : {fp32_acc:.2f}%")
print(f"INT8 Accuracy : {int8_acc:.2f}%")

print(f"HA Accuracy   : {ha_acc:.2f}%")
print(f"MA Accuracy   : {ma_acc:.2f}%")
print(f"LA Accuracy   : {la_acc:.2f}%")
print(f"FA Accuracy   : {fa_acc:.2f}%")

print("================================================")
