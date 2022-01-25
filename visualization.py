import torch
from torchviz import make_dot

from src.models import *
from src.data import Dataset


def main():
    dataset = Dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1)
    models = [Generator(), GlobalDiscriminator(), LocalDiscriminator()]

    for model in models[:2]:
        for batch in dataloader:
            pred = model(batch[0])
            break
        make_dot(pred, params=dict(list(model.named_parameters()))).render(type(model).__name__, format="png")

if __name__ == '__main__':
    main()