import torchvision
import torch


class MNISTDataLoader:
    def __init__(self, args):
        image_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        # image datasets
        train_dataset = torchvision.datasets.MNIST(
            args.dir, train=True, download=True, transform=image_transform
        )
        test_dataset = torchvision.datasets.MNIST(
            args.dir, train=False, download=True, transform=image_transform
        )
        # data loaders
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size_train, shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size_test, shuffle=True
        )
