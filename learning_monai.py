import os
import shutil
import tempfile
import matplotlib.pyplot as plt
import PIL
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import classification_report

from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
)
from monai.utils import set_determinism

class classifier_dataset(torch.utils.data.Dataset):
        def __init__(self, image_files, labels, transforms):
            self.image_files = image_files
            self.labels = labels
            self.transforms = transforms

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, index):
            return self.transforms(self.image_files[index]), self.labels[index]

if __name__ == "__main__":
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory
    print(root_dir)
    training_dir = r"C:\Users\jaime\OneDrive\Downloads\archive (1)\MU-Diff images\processed\train"
    val_dir = r"C:\Users\jaime\OneDrive\Downloads\archive (1)\MU-Diff images\processed\val"
    test_dir = r"C:\Users\jaime\OneDrive\Downloads\archive (1)\MU-Diff images\processed\test"
    class_names = ['healthy','tumour']
    num_class = len(class_names)
    training_image_files = [
        [os.path.join(training_dir, class_names[i], x) for x in os.listdir(os.path.join(training_dir, class_names[i]))]
        for i in range(num_class)
    ]
    val_image_files = [
        [os.path.join(val_dir, class_names[i], x) for x in os.listdir(os.path.join(val_dir, class_names[i]))]
        for i in range(num_class)
    ]
    test_image_files = [
        [os.path.join(test_dir, class_names[i], x) for x in os.listdir(os.path.join(test_dir, class_names[i]))]
        for i in range(num_class)
    ]
    num_training = [len(training_image_files[i]) for i in range(num_class)]
    num_val = [len(val_image_files[i]) for i in range(num_class)]
    num_test = [len(test_image_files[i]) for i in range(num_class)]
    train_x = []
    val_x = []
    test_x = []
    train_y = []
    val_y = []
    test_y = []
    for i in range(num_class):
        train_x.extend(training_image_files[i])
        train_y.extend([i] * num_training[i])
    for i in range(num_class):
        val_x.extend(val_image_files[i])
        val_y.extend([i] * num_val[i])
    for i in range(num_class):
        test_x.extend(test_image_files[i])
        test_y.extend([i] * num_test[i])
    num_total = len(train_y)
    image_width, image_height = PIL.Image.open(train_x[0]).size

    print(f"Total image count: {num_total}")
    print(f"Image dimensions: {image_width} x {image_height}")
    print(f"Label names: {class_names}")
    print(f"Label counts: {num_training}")
    print(f"Training count: {len(train_x)}, Validation count: " f"{len(val_x)}, Test count: {len(test_x)}")

    train_transforms = Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            ScaleIntensity(),
            RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        ]
    )

    val_transforms = Compose([LoadImage(image_only=True), EnsureChannelFirst(), ScaleIntensity()])

    y_pred_trans = Compose([Activations(softmax=True)])
    y_trans = Compose([AsDiscrete(to_onehot=num_class)])

    train_ds = classifier_dataset(train_x, train_y, train_transforms)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=8)

    val_ds = classifier_dataset(val_x, val_y, val_transforms)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=8)

    test_ds = classifier_dataset(test_x, test_y, val_transforms)
    test_loader = DataLoader(test_ds, batch_size=32, num_workers=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=num_class).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    max_epochs = 4
    val_interval = 1
    auc_metric = ROCAUCMetric()

    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    writer = SummaryWriter()

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
            epoch_len = len(train_ds) // train_loader.batch_size
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images, val_labels = (
                        val_data[0].to(device),
                        val_data[1].to(device),
                    )
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)
                y_onehot = [y_trans(i) for i in decollate_batch(y, detach=False)]
                y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
                auc_metric(y_pred_act, y_onehot)
                result = auc_metric.aggregate()
                auc_metric.reset()
                del y_pred_act, y_onehot
                metric_values.append(result)
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                if result > best_metric:
                    best_metric = result
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current AUC: {result:.4f}"
                    f" current accuracy: {acc_metric:.4f}"
                    f" best AUC: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
                writer.add_scalar("val_accuracy", acc_metric, epoch + 1)

    print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")
    writer.close()

    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val AUC")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.show()

    model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth"), weights_only=True))
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels = (
                test_data[0].to(device),
                test_data[1].to(device),
            )
            pred = model(test_images).argmax(dim=1)
            for i in range(len(pred)):
                y_true.append(test_labels[i].item())
                y_pred.append(pred[i].item())

    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    if directory is None:
        shutil.rmtree(root_dir)