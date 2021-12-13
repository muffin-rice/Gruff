## Sample: 

Install Jax, Jaxlib, Haiku, and add pyspiel.cc to site-packages

```shell
cd samples
supervised bridge_supervised_learning.py --data_path=data/
```

## Supervised

Our supervised learning uses PyTorch Lightning located in supervised.   

## Reinforcement Learning 

We use pretrained models from supervised before utilizing reinforcement learning.

Scripts are located in a2c/, adversarial/, and dqn/; run the scripts while specifying -e (episodes) and -pt (pretrained model location) arguments. 