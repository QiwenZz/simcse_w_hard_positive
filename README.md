## Simple Contrastive Learning of Sentence Embeddings with Hard Positive

This repo is based on SimCSE with the hard positive trick. The code is all from the official SimCSE repo. Only two files are changed to implement the hard positive feature: 1. train.py 2. models.py.

Hard positive is a way to find more meaningful positive pairs that teach the model to learn a better representation based on our research.

See our report for more insights: [Contrastive Learning with Hard Positive Samples in NLP tasks](https://github.com/QiwenZz/simcse_w_hard_positive/blob/main/Contrastive%20Learning%20with%20Sub-optimal%20Positive%20Samples%20in%20NLP%20tasks.pdf)

To train your SimCSE model, edit and run run_unsup_example.sh.

In run_unsup_example.sh, you can choose whether to train the model with hard positive selection with the hyper-parameter `--use_hard_positive` and the number of positive sample candidates with `hard_positive_candidates`.

Currently, the code only supports unsupervised training with txt file as the input training data. 


