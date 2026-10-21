"""Pipeline Imagenette 160 px untuk Google Colab."""
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

def get_imagenette_dataloaders(
    data_dir="/content/imagenette2-160",
    img_size=160,
    batch_size=64,
    val_split=0.1,
    seed=0,
    num_workers=2,
):
    data_dir = Path(data_dir)
    train_dir, val_dir = data_dir / "train", data_dir / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di {data_dir}. "
            "Jalankan sel unduh/ekstrak Imagenette di Colab terlebih dahulu."
        )

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Dataset sudah diekstrak ke folder train/ dan val/: gunakan ImageFolder.
    train_aug = datasets.ImageFolder(train_dir, transform=train_transform)
    train_clean = datasets.ImageFolder(train_dir, transform=eval_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)

    if train_aug.classes != train_clean.classes or train_aug.classes != val_dataset.classes:
        raise ValueError("Urutan kelas train dan val tidak konsisten.")

    indices = list(range(len(train_aug)))
    targets = train_aug.targets  # API publik ImageFolder; bukan ._samples.
    train_idx, internal_val_idx = train_test_split(
        indices,
        test_size=val_split,
        random_state=seed,
        stratify=targets,
    )

    train_subset = Subset(train_aug, train_idx)
    internal_val_subset = Subset(train_clean, internal_val_idx)
    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    train_loader = DataLoader(train_subset, shuffle=True, **kwargs)
    internal_val_loader = DataLoader(internal_val_subset, shuffle=False, **kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **kwargs)
    return train_loader, internal_val_loader, val_loader, train_aug.classes

if __name__ == "__main__":
    train_loader, internal_val_loader, val_loader, classes = get_imagenette_dataloaders()
    print("Jumlah kelas:", len(classes))
    print("Kelas:", classes)
    print("Data train:", len(train_loader.dataset))
    print("Validasi internal:", len(internal_val_loader.dataset))
    print("Evaluasi akhir:", len(val_loader.dataset))
    images, labels = next(iter(train_loader))
    print("Batch train:", tuple(images.shape))
    print("Batch label:", tuple(labels.shape))
    assert images.ndim == 4 and images.shape[1:] == (3, 160, 160)
    assert len(classes) == 10
    print("Pipeline berhasil divalidasi.")
