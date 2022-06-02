# Anomaly_Detection_Comparison

A comparison of three anomaly detection methods: [SPE]([https://arxiv.org/pdf/1909.03500v3.pdf](https://github.com/ZhiningLiu1998/self-paced-ensemble)), [a contrastive learning method](https://openreview.net/forum?id=_hszZbt46bT), and [Devnet](https://github.com/GuansongPang/deviation-network) on [Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset.

This respository includes the implementation of the first two methods. Due to the conflicts of the dependencies, the code of Devnet is in [this respository](https://github.com/odilecooper/deviation-network#devnet-an-end-to-end-anomaly-score-learning-network).

## Usage

To train and test the models on the same data that Devnet use, which is the normalised version of Fraud dataset:
```python
python main.py --use_normalised
```

To run on the original Fraud dataset:
```python
python main.py
```

## Settings

Model parameters are set to be the same as stated in the papers. For SPE, the number of estimators isn't mentioned in the paper, thus choose 289 estimators to maximize the number of samples in training. In this way the SPE method uses roughly as many training data as the other two method use.

Use 80% of the data in training and 20% in testing. Anomalies are not ruled out from training data.


## Results
| method      | AUCROC | AUCPR  |
| ----------- | ------ | ------ |
| SPE         | 0.9387 | 0.7625 |
| DevNet      | 0.9818 | 0.7016 |
| contrastive | 0.8402 | 0.0497 |

Experiment results are the mean results of ten runs for each method.
