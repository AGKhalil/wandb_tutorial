# Installation

```
conda create -n wandb_tutorial python==3.8
conda activate wandb_tutorial
pip install -r requirements.txt
```

# Quick usage

1. Create a `wandb` account through this [link](https://wandb.auth0.com/login?state=hKFo2SBMTmdzNXY4dkRILS1qSGdIUzNjUEE2UHAyT081N240RaFupWxvZ2luo3RpZNkgTXd0MVlLSWhLcENNWXRBRG1HM1Rjc0lqWkY5RkU4REujY2lk2SBWU001N1VDd1Q5d2JHU3hLdEVER1FISUtBQkhwcHpJdw&client=VSM57UCwT9wbGSxKtEDGQHIKABHppzIw&protocol=oauth2&nonce=dHRUVXJDbDF1c0c5alFtXw%3D%3D&redirect_uri=https%3A%2F%2Fapi.wandb.ai%2Foidc%2Fcallback&response_mode=form_post&response_type=id_token&scope=openid%20profile%20email&signup=true).

2. Go into your chosen example directory

```
cd coin_toss
```

or

```
cd CNN-MNIST
```

3. Edit the `train.py` script: replace `ENTITY` by your `wandb` username.

```
wandb.init(
    entity="ENTITY",
    project="wandb_tutorial",
    config=args,
    save_code=True,
)
```

4. Run the `train.py` script with default or your chosen arguments. `coin_toss` example:

```
python train.py --prob 0.75
```

# Use `wandb` in 5 easy steps

1. Create a `wandb` [account](https://wandb.auth0.com/login?state=hKFo2SBMTmdzNXY4dkRILS1qSGdIUzNjUEE2UHAyT081N240RaFupWxvZ2luo3RpZNkgTXd0MVlLSWhLcENNWXRBRG1HM1Rjc0lqWkY5RkU4REujY2lk2SBWU001N1VDd1Q5d2JHU3hLdEVER1FISUtBQkhwcHpJdw&client=VSM57UCwT9wbGSxKtEDGQHIKABHppzIw&protocol=oauth2&nonce=dHRUVXJDbDF1c0c5alFtXw%3D%3D&redirect_uri=https%3A%2F%2Fapi.wandb.ai%2Foidc%2Fcallback&response_mode=form_post&response_type=id_token&scope=openid%20profile%20email&signup=true).
2. Install `wandb` commandline tool.

```
pip install wandb
```

3. Import `wandb`

```
import wandb
```

4. Initialize `wandb`, replacing `ENTITY` with your `wandb` account username.

```
wandb.init(
    entity="ENTITY",
    project="wandb_tutorial",
    config=args,
    save_code=True,
)
```

5. Log to `wandb`

```
wandb.log(
    {
        "train/loss": loss,
    }
)
```

# FAQs

**What if I'm already logging to tensorboard?**
`wandb` can automatically log your tensorboard metrics. All you need to do is add the `sync_tensorboard` flag to the `wandb` initialization. Example:

```
wandb.init(
    entity="ENTITY",
    project="wandb_tutorial",
    config=args,
    save_code=True,
    sync_tensorboard=True,
)
```

**What if I run wandb in a notebook?**
Everything will work nicely but you need to add `wandb.finish()` at the very bottom of your notebook (after you're done with training/evaluation.)

**Will `wandb` work with my code?**
`wandb` is framework agnostic. You can add it yourself as we demonstrate in this tutorial or rely on `wandb` [integrations](https://docs.wandb.ai/guides/integrations).

For instance, to use `wandb` with `HuggingFace`, you need only [do](https://docs.wandb.ai/guides/integrations/huggingface):

```
from transformers import TrainingArguments, Trainer

args = TrainingArguments(... , report_to="wandb")
trainer = Trainer(... , args=args)
```
