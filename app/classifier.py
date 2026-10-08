"""Image classification with the Assignment 2 CNN (trained on CIFAR-10)."""
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from helper_lib.model import CIFAR10_CLASSES, get_model

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "cnn_classifier.pth"


class ImageClassifier:
    def __init__(self, weights_path=DEFAULT_WEIGHTS):
        self.model = get_model("AssignmentCNN")
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.eval()  # inference mode
        # Must match the preprocessing used in training: resize to 64x64, scale to [0, 1].
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    def predict(self, image: Image.Image, top_k: int = 3) -> dict:
        """Return the predicted CIFAR-10 class and the top_k class probabilities."""
        x = self.transform(image.convert("RGB")).unsqueeze(0)  # shape (1, 3, 64, 64)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(x), dim=1)[0]
        top = torch.topk(probabilities, top_k)
        predictions = [
            {"label": CIFAR10_CLASSES[int(i)], "probability": round(float(p), 4)}
            for p, i in zip(top.values, top.indices)
        ]
        return {
            "predicted_class": predictions[0]["label"],
            "confidence": predictions[0]["probability"],
            "top_predictions": predictions,
        }

    def describe(self) -> dict:
        return {
            "architecture": str(self.model),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "input": "RGB image, resized to 64x64",
            "classes": CIFAR10_CLASSES,
        }
