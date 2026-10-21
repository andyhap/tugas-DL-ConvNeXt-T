
from pathlib import Path

import torch
from torchinfo import summary

from convnext_tiny_imagenette import build_model


def main():
    model = build_model(num_classes=10)
    model.eval()

    # Input dummy: satu citra RGB berukuran 160 x 160.
    dummy_input = torch.randn(1, 3, 160, 160)

    with torch.no_grad():
        output = model(dummy_input)

    total_params = sum(
        parameter.numel() for parameter in model.parameters()
    )
    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Input shape          :", tuple(dummy_input.shape))
    print("Output shape         :", tuple(output.shape))
    print("Total parameters     :", f"{total_params:,}")
    print("Trainable parameters :", f"{trainable_params:,}")

    assert output.shape == (1, 10), (
        f"Output seharusnya (1, 10), tetapi mendapat {tuple(output.shape)}"
    )

    print("\nPengujian bentuk tensor berhasil.\n")

    # Buat ringkasan model aktual.
    model_summary = summary(
        model,
        input_size=(1, 3, 160, 160),
        col_names=[
            "input_size",
            "output_size",
            "num_params",
            "trainable",
        ],
        depth=4,
        verbose=0,
    )

    summary_text = str(model_summary)

    # menyimpan ringkasan model ke file teks
    output_path = Path(__file__).resolve().parent / "model_summary.txt"
    output_path.write_text(summary_text + "\n", encoding="utf-8")

    print(f"Ringkasan model disimpan di: {output_path}")


if __name__ == "__main__":
    main()
