"""Tes bentuk input/output model dan simpan ringkasan arsitektur."""
from pathlib import Path
import torch
from torchinfo import summary
from src.model import build_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=10).to(device).eval()
    dummy = torch.randn(1, 3, 160, 160, device=device)
    with torch.no_grad():
        output = model(dummy)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Device               :", device)
    if device.type == "cuda":
        print("GPU                  :", torch.cuda.get_device_name(0))
    print("Input shape          :", tuple(dummy.shape))
    print("Output shape         :", tuple(output.shape))
    print("Total parameters     :", f"{total_params:,}")
    print("Trainable parameters :", f"{trainable_params:,}")
    assert output.shape == (1, 10)

    model_summary = summary(
        model,
        input_size=(1, 3, 160, 160),
        device=str(device),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        depth=4,
        verbose=0,
    )
    output_path = Path(__file__).resolve().parent / "model_summary.txt"
    output_path.write_text(str(model_summary) + "\n", encoding="utf-8")
    print("Ringkasan disimpan:", output_path)
    print("Tes model berhasil.")

if __name__ == "__main__":
    main()
