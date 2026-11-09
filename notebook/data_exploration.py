from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

#dataset paths 
project_root = Path(__file__).resolve().parent.parent
dataset_path = project_root / "brain_mri"
train_path = dataset_path / "train"
val_path = dataset_path / "val"
test_path = dataset_path / "test"

#class distribution 
image_extensions = {".jpg", ".jpeg", ".png"}
for class_folder in train_path.iterdir():

    if class_folder.is_dir():

        images = [
            file for file in class_folder.iterdir()
            if file.suffix.lower() in image_extensions
        ]

        print(class_folder.name, ":", len(images))

#class distribution visualization
classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
counts = [4388, 3292, 4880, 3035]

plt.figure(figsize=(8, 5))

plt.bar(classes, counts)

plt.title("Training Dataset Class Distribution")
plt.xlabel("Brain Tumor Classes")
plt.ylabel("Number of Images")

plt.show()

#sample image visualization 
classes = ["glioma", "meningioma", "notumor", "pituitary"]
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for i, class_name in enumerate(classes):

    class_folder = train_path / class_name

    image_files = list(class_folder.glob("*"))

    image_path = image_files[0]
    image = Image.open(image_path)

    axes[i].imshow(image)
    axes[i].set_title(class_name)
    axes[i].axis("off")

plt.tight_layout()
plt.show()


#image dimesions and color modes 
for class_name in classes:

    class_folder = train_path / class_name

    image_files = list(class_folder.glob("*"))

    image = Image.open(image_files[0])

    print(f"{class_name}:")
    print("Dimensions:", image.size)
    print("Color mode:", image.mode)
    print()

