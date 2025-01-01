# OCR-CFT
The official code of [Optimistic Critic Reconstruction and Constrained Fine-Tuning for General Offline-to-Online RL](https://arxiv.org/abs/2412.18855).
## Acknowledgments

This project makes use of the following open-source projects:

- **[CORL](https://github.com/tinkoff-ai/CORL)**: Implementation of the offline training process.
- **[Uni-O4](https://github.com/Lei-Kun/Uni-O4)**: Implementation of PPO.


***
## Install
For installation instructions, please refer to the [CORL](https://github.com/tinkoff-ai/CORL) repository for detailed guidance.
***

## Run
Take O2SAC from the results of CQL as an example.
### offline pre-train
To run the offline pre-training, use the following command:
```
cd offline
python cql.py --env hopper-medium-v2 --seed 0
```

### online fine-tuning
To perform online fine-tuning, use the following command:
```
cd finetune
python O2SAC.py --env hopper-medium-v2 --seed 0
```
