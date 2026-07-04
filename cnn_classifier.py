# ----- Imports ------

import os
import csv

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from string import ascii_lowercase

# ----- Loading the dataset ------

# Setting the path of the training dataset (that was already provided to you)
running_local = True if os.getenv('JUPYTERHUB_USER') is None else False
DATASET_PATH = "."

# Set the location of the dataset
if running_local:
    # If running on your local machine, the sign_lang_train folder's path should be specified here
    local_path = "sign_lang_train"
    if os.path.exists(local_path):
        DATASET_PATH = local_path
else:
    # If running on the Jupyter hub, this data folder is already available
    # You DO NOT need to upload the data!
    DATASET_PATH = "/data/mlproject22/sign_lang_train"

# Utility function
def read_csv(csv_file):
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

# ------ Data loading using PyTorch ------

class SignLangDataset(Dataset):
    """Sign language dataset"""

    def __init__(self, csv_file, root_dir, class_index_map=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.class_index_map = class_index_map
        self.transform = transform
        # List of class names in order
        self.class_names = list(map(str, list(range(10)))) + list(ascii_lowercase)

    def __len__(self):
        """
        Calculates the length of the dataset-
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns one sample (dict consisting of an image and its label)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Read the image and labels
        image_path = os.path.join(self.root_dir, self.data[idx][1])
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Shape of the image should be H,W,C where C=1
        image = np.expand_dims(image, 0)
        # The label is the index of the class name in the list ['0','1',...,'9','a','b',...'z']
        # because we should have integer labels in the range 0-35 (for 36 classes)
        label = self.class_names.index(self.data[idx][0])

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

# ----- CNN Classifier ------

class CNNClassifier():
    """
    CNN classifier
    """

    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """

        # Load dataset
        self.dataset = SignLangDataset(csv_file, root_dir)
        self.dataset_size = self.dataset.__len__()

        train_size = int(self.dataset_size * 0.70)
        test_size = int(self.dataset_size * 0.15)
        val_size = self.dataset_size - train_size - test_size

        # Split data
        self.train_split, self.val_split, self.test_split = random_split(
            self.dataset, [train_size, val_size, test_size]
        )

        # Wrap in loaders to feed model
        self.train_loader = torch.utils.data.DataLoader(
            self.train_split,
            batch_size=128,
            shuffle=True,
            num_workers=12,
            persistent_workers=True,
            pin_memory = True
        )

        self.train_transform = transforms.Compose([
            transforms.RandomRotation(10, fill=255),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=255)
        ])

        # Transforms images randomly during training so the model sees slightly different versions, forcing it to learn general features
        self.test_loader = torch.utils.data.DataLoader(
            self.test_split,
            batch_size=128,
            shuffle=False,
            num_workers=12,
            persistent_workers=True,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            self.val_split,
            batch_size=128,
            shuffle=False,
            num_workers=12,
            persistent_workers=True,
            pin_memory=True
        )

        # Sequential is a container that pumps input through given layers in order
        self.model = nn.Sequential()

        # Try to make it GPU accelerated
        self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    def build_model(self):
        """
        Builds the CNN model
        """

        # Workflow:
            # 1. Cov2d extracts basic edges
            # 2. BatchNorm2d TODO
            # 3. ReLu adds non-linearity
            # 4. MaxPool2d shrinks size in half
                # Repeating this 3 times for 3 layers

            # 5. Flatten turns 2D grid into 1D vector

            # 6. Linear outputs 36 class scores
            # 7. ReLu again
            # 8. Dropout TODO
                # Repeating this 2 times for 2 fully connected layers

            # 9. Linear again

        # Layer 1
        self.model.add_module("Conv2d-1", nn.Conv2d(1, 32, 3, padding=1))
        self.model.add_module("BatchNorm2d-1", nn.BatchNorm2d(32))  # Match Conv2d-1 (32)
        self.model.add_module("ReLU-1", nn.ReLU())
        self.model.add_module("MaxPool2d-1", nn.MaxPool2d(2, 2))

        # Layer 2
        self.model.add_module("Conv2d-2", nn.Conv2d(32, 64, 3, padding=1))
        self.model.add_module("BatchNorm2d-2", nn.BatchNorm2d(64))  # Match Conv2d-2 (64)
        self.model.add_module("ReLU-2", nn.ReLU())
        self.model.add_module("MaxPool2d-2", nn.MaxPool2d(2, 2))

        # Layer 3
        self.model.add_module("Conv2d-3", nn.Conv2d(64, 128, 3, padding=1))
        self.model.add_module("BatchNorm2d-3", nn.BatchNorm2d(128))  # Match Conv2d-3 (128)
        self.model.add_module("ReLU-3", nn.ReLU())
        self.model.add_module("MaxPool2d-3", nn.MaxPool2d(2, 2))

        self.model.add_module("Flatten", nn.Flatten())

        # Fully connected layer 1
        self.model.add_module("Linear-1", nn.Linear(128 * 16 * 16, 256))
        self.model.add_module("ReLU-4", nn.ReLU())
        self.model.add_module("Dropout-1", nn.Dropout(p=0.15))

        # Fully connected layer 2
        self.model.add_module("Linear-2", nn.Linear(256, 128))
        self.model.add_module("ReLU-5", nn.ReLU())
        self.model.add_module("Dropout-2", nn.Dropout(p=0.15))

        self.model.add_module("Linear-3", nn.Linear(128, 36))

        self.model.to(self.device)

    def train(self):
        """
        Trains the CNN model
        """

        print(f"Training is accelerating on: {self.device}")

        loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

        best_loss = float("inf")
        patience, patience_counter = 10, 0

        for epoch in range(100):

            self.model.train()
            running_loss = 0

            for batch in self.train_loader:
                # Reset
                optimizer.zero_grad(set_to_none=True)

                # Unpack data from batch
                image = batch['image'].float()
                image = torch.stack([self.train_transform(img) for img in image])
                image = image.to(self.device)
                label = batch['label'].to(self.device)

                # Forward pass
                predictions = self.model(image)

                # Calculate loss
                loss = loss_function(predictions, label)

                # Backpropagation
                loss.backward()

                # Accumulated epoch loss
                running_loss += loss.item()

                # Optimize
                optimizer.step()

            epoch_loss = running_loss / len(self.train_loader)

            # Measure loss on the test set to get an honest signal of how well the model generalizes to data it has never trained on
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in self.val_loader:
                    image = batch['image'].float().to(self.device)
                    label = batch['label'].to(self.device)
                    predictions = self.model(image)
                    val_loss += loss_function(predictions, label).item()
            val_loss /= len(self.val_loader)

            print(f"Epoch [{epoch + 1}/100] - Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}")

            scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pth")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

        self.model.load_state_dict(torch.load("best_model.pth"))
        print("Loaded best model weights")

    def predict(self, image):
        """
        Predicts the class for a given image
        """

        # Set model to evaluation mode
        self.model.eval()

        # Turns off tracking (saves memory)
        with torch.no_grad():

            # Model expects batch, so convert the image
            batch_image = image.float().unsqueeze(0)
            predictions = self.model(batch_image)

            winner = torch.argmax(predictions, 1)

            predicted_char = self.dataset.class_names[winner.item()]

        return predicted_char

    def test(self):
        """
        Tests the CNN model
        """

        self.model.eval()

        # Initialize scoreboard trackers
        correct_guesses = 0
        total_images = 0

        with torch.no_grad():
            for batch in self.test_loader:
                image = batch['image'].float().to(self.device)
                label = batch['label'].to(self.device)

                predictions = self.model(image)

                winner = torch.argmax(predictions, dim=1)

                # Add the number of images in this batch to our total counter
                total_images += label.size(0)

                # Compare predictions to actual labels and sum up the correct ones
                correct_guesses += (winner == label).sum().item()

        # Calculate the final accuracy
        accuracy = (correct_guesses / total_images) * 100
        print(f"Accuracy: {accuracy:.2f}%")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

# ------ Testing ------

if __name__ == '__main__':
    # Setup
    cnn_classifier = CNNClassifier("labels.csv", DATASET_PATH)
    cnn_classifier.build_model()

    # If model was not trained yet:
    # classifier.train()

    # If model was already trained:
    cnn_classifier.load_model("best_model.pth")

    # Sanity check
    cnn_classifier.test()