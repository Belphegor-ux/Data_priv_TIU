import foolbox as fb
import torch
import torchvision.models as models
import numpy as np


model = models.resnet18(pretrained=True).eval()
preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


fmodel = fb.PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)


images = np.load('test_images.npy')
labels = np.load('test_labels.npy')


images = torch.from_numpy(images).to(torch.float32)
labels = torch.from_numpy(labels).to(torch.int64)

attack = fb.attacks.FGSM()
epsilons = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]
raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilons)


robust_accuracy = 1 - success.float().mean(axis=-1)
for eps, acc in zip(epsilons, robust_accuracy):
    print(f"Epsilon: {eps:<8} Robust accuracy: {acc.item():.4f}")
