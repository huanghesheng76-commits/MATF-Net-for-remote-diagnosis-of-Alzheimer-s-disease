import matplotlib.pyplot as plt
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from my_dataset_new import MyDataSet
from model_eff2 import EfficientNet
from sklearn.metrics import accuracy_score, f1_score


def run():
    num_epochs = 4
    save_dir = "../pth/Our_AD_NC_test/"

    train_data_path = "/home/huang/HHS/ADNI_Dataset/ADNI1_1/train"
    val_data_path = "/home/huang/HHS/ADNI_Dataset/ADNI1_1/val"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = EfficientNet(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        num_classes=2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.8, 1.2),
            shear=10
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = MyDataSet(
        train_data_path,
        images_class=["P-AD", "P-NC"],
        transform=train_transform
    )

    val_dataset = MyDataSet(
        val_data_path,
        images_class=["P-AD", "P-NC"],
        transform=eval_transform
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=1,
        drop_last=False
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=1,
        drop_last=False
    )

    def train(
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        device=device
    ):
        training_history = {
            "train_loss": [],
            "valid_loss": [],
            "accuracy": [],
            "f1_macro": []
        }

        for epoch in range(num_epochs):
            train_loss = 0.0
            val_loss = 0.0

            model.train()

            for batch_images, batch_labels in tqdm(train_loader):
                batch_images = [
                    image.to(device) for image in batch_images
                ]
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad()

                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_labels.size(0)

            model.eval()

            val_predictions = []
            val_targets = []

            with torch.no_grad():
                for batch_images, batch_labels in val_loader:
                    batch_images = [
                        image.to(device) for image in batch_images
                    ]
                    batch_labels = batch_labels.to(device)

                    outputs = model(batch_images)
                    loss = criterion(outputs, batch_labels)

                    predictions = (
                        torch.argmax(outputs, dim=1).cpu().numpy()
                    )
                    targets = batch_labels.cpu().numpy()

                    val_predictions.extend(predictions)
                    val_targets.extend(targets)

                    val_loss += loss.item() * batch_labels.size(0)

            val_loss /= len(val_loader.dataset)

            val_accuracy = accuracy_score(
                val_targets,
                val_predictions
            )
            val_f1_macro = f1_score(
                val_targets,
                val_predictions,
                average="macro"
            )

            print(
                f"Epoch: {epoch + 1} "
                f"Train Loss: {train_loss:.3f} "
                f"Valid Loss: {val_loss:.3f} "
                f"ACC: {val_accuracy:.3f} "
                f"F1_MACRO: {val_f1_macro:.3f}"
            )

            training_history["train_loss"].append(train_loss)
            training_history["valid_loss"].append(val_loss)
            training_history["accuracy"].append(val_accuracy)
            training_history["f1_macro"].append(val_f1_macro)

            with open(
                os.path.join(save_dir, "ADNI1_4_AD_NC_Epoch1.txt"),
                "a"
            ) as log_file:
                log_file.write(
                    f"Epoch: {epoch + 1} "
                    f"Train Loss: {train_loss:.3f} "
                    f"Valid Loss: {val_loss:.3f} "
                    f"ACC: {val_accuracy:.3f} "
                    f"F1_MACRO: {val_f1_macro:.3f}\n"
                )

        return training_history

    training_history = train(
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader
    )

    figure, axes = plt.subplots(2, 1, figsize=(10, 6))

    axes[0].plot(
        training_history["train_loss"],
        label="Train Loss"
    )
    axes[0].plot(
        training_history["valid_loss"],
        label="Valid Loss"
    )
    axes[0].set_title("Losses")
    axes[0].legend()

    axes[1].plot(
        training_history["accuracy"],
        label="Accuracy"
    )
    axes[1].plot(
        training_history["f1_macro"],
        label="F1-macro"
    )
    axes[1].set_title("Classification metrics")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "train_stats.png"))
    plt.show()


if __name__ == "__main__":
    run()
