# This repo's code is adapted from https://github.com/jiuntian/pytorch-mnist-example/blob/master/pytorch-mnist.ipynb

import argparse
import numpy as np
import wandb
import torch
import torch.nn.functional as F
from dataloader import MNISTDataLoader
from model import CNN
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=np.random.randint(0, 1000, 1)[0], type=int)
    parser.add_argument("--n_epochs", default=10, type=int)
    parser.add_argument("--log_interval", default=2000, type=int)
    # parser.add_argument("--model_save_freq", default=100000, type=int)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str
    )

    # dataloader args
    parser.add_argument("--dir", default="./data", type=str)
    parser.add_argument("--batch_size_train", default=64, type=int)
    parser.add_argument("--batch_size_test", default=1024, type=int)

    # optimizer params
    parser.add_argument("--learning_rate", default=0.01, type=float)
    parser.add_argument("--momentum", default=0.5, type=float)

    return parser.parse_args()


def main():
    args = parse_args()
    wandb.init(
        entity="agkhalil",
        project="wandb_tutorial",
        config=args,
        save_code=True,
        sync_tensorboard=True,
    )

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    dataloader = MNISTDataLoader(args)
    model = CNN().to(device)
    wandb.watch(model)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
    )

    _step = 0
    for _ in range(args.n_epochs):
        # train
        model.train()
        counter = 0
        for (data, target) in tqdm(dataloader.train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/accuracy": correct / pred.shape[0],
                },
            )
            counter += 1
            _step += 1

        # eval
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in dataloader.test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(dataloader.test_loader.dataset)

        table = wandb.Table(
            columns=["Image", "Prediction"],
            data=[
                [wandb.Image(img), prediction.item()]
                for img, prediction in zip(
                    torch.stack([data[:10]]).squeeze(0), pred[:10]
                )
            ],
        )

        wandb.log(
            {
                "eval/loss": test_loss,
                "eval/accuracy": correct / len(dataloader.test_loader.dataset),
                "eval/predictions": table,
            },
        )

        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(dataloader.test_loader.dataset),
                100.0 * correct / len(dataloader.test_loader.dataset),
            )
        )


if __name__ == "__main__":
    main()
