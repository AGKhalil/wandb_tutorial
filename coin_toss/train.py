# This repo's code is adapted from https://colab.research.google.com/drive/1ncNQZzF44_h-3nlrM7enIjhnxXzmwb1B?usp=sharing

import random
from collections import defaultdict
import argparse
from tqdm import tqdm
import wandb
import numpy as np


def coin_toss(prob, head, tail):
    r = random.uniform(0, 1)
    if r >= prob:
        return tail
    return head


def run_coin_toss_experiment(num_tosses: int, prob, head, tail):
    exp_results = []
    for i in range(num_tosses):
        exp_results.append(coin_toss(prob=prob, tail=tail, head=head))
    return exp_results


def run_simulation(
    args,
    logger=None,
):
    counter_ = defaultdict(float)
    for exp_i in tqdm(range(args.num_experiments)):
        result_ = tuple(
            run_coin_toss_experiment(
                num_tosses=args.tosses_per_experiment,
                head=args.head,
                tail=args.tail,
                prob=args.prob,
            )
        )
        counter_[result_] += 1
        if exp_i % args.log_interval == 0:
            for k, v in counter_.items():
                logger.log({"".join(k): v / args.num_experiments})

    if args.normalize:
        for k, v in counter_.items():
            counter_[k] = v / args.num_experiments

    return counter_


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=np.random.randint(0, 1000, 1)[0], type=int)
    parser.add_argument("--prob", default=0.5, type=float)
    parser.add_argument("--normalize", default=True, type=bool)
    parser.add_argument("--tosses_per_experiment", default=2, type=int)
    parser.add_argument("--num_experiments", default=int(1e7), type=int)
    parser.add_argument("--head", default="h", type=str)
    parser.add_argument("--tail", default="t", type=str)
    parser.add_argument("--log_interval", default=1e4, type=int)

    return parser.parse_args()


def main():
    args = parse_args()
    wandb.init(
        entity="agkhalil",
        project="wandb_tutorial",
        config=args,
        save_code=True,
    )

    random.seed(args.seed)
    results = run_simulation(
        args,
        logger=wandb,
    )
    for k, v in results.items():
        print(k, v)


if __name__ == "__main__":
    main()
