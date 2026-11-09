import os
os.environ["KERAS_BACKEND"] = "torch"
from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import keras
from collections import Counter
import matplotlib.pyplot as plt

def convert_rgb(img):
    return img.convert("RGB")


def to_channels_last(img):
    return img.permute(1, 2, 0)

    # Training code here
project_root = Path(__file__).resolve().parent.parent
dataset_path = project_root / "brain_mri"
train_path = dataset_path / "train"
test_path = dataset_path / "test"
val_path = dataset_path / "val"

train_transform = transforms.Compose([
    transforms.Lambda(convert_rgb),
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),
    transforms.ToTensor(),
    transforms.Lambda(to_channels_last)
])

val_test_transform = transforms.Compose([
    transforms.Lambda(convert_rgb),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(to_channels_last)
])

train_data = datasets.ImageFolder(
    root=train_path,
    transform=train_transform
)

train_loader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

val_data = datasets.ImageFolder(
    root= val_path,
    transform=val_test_transform
)
val_loader = DataLoader(
    val_data,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

test_data = datasets.ImageFolder(
    root=  test_path,
    transform= val_test_transform
)
test_loader = DataLoader(
    test_data,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)
if __name__ == "__main__":

    print("Classes:", train_data.classes)
    print("Class mapping:", train_data.class_to_idx)

    images, labels = next(iter(train_loader))

    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Minimum pixel value:", images.min().item())
    print("Maximum pixel value:", images.max().item())

    print("Training images:", len(train_data))
    print("Validation images:", len(val_data))
    print("Testing images:", len(test_data))

    print("Training:", train_data.class_to_idx)
    print("Validation:", val_data.class_to_idx)
    print("Testing:", test_data.class_to_idx)

    datasets_list = {
        "Training": train_data,
        "Validation": val_data,
        "Testing": test_data
    }

    for name, dataset in datasets_list.items():

        label_counts = Counter(dataset.targets)

        print(f"\n{name} Distribution:")

        for class_name, class_index in dataset.class_to_idx.items():
            print(class_name, ":", label_counts[class_index])

    fig, axes = plt.subplots(1, 5, figsize=(15, 4))

    for i in range(5):
        image, label = train_data[0]

        axes[i].imshow(image)
        axes[i].set_title(f"Version {i+1}")
        axes[i].axis("off")

    plt.show()

