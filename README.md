
# Abstract to Title Generation

This project is based on a bachelor [thesis](https://github.com/Xieyichen/Thesis) that aims on automatically generating titles from paper abstracts. This is done using different generator models (bart-base, bart-cnn, bart-xsum, gpt2, t5 and pegasus-xsum) to generate titles, a reward model to score titles trained on human annotations and a reinforcement learning environment to optimize that model for better text generation.

We improve the reward model with further human annotations and introduce and additional objective of generating humor. Therefor we annotate scientific humor datasets and created a pipelined learning for learning a joint reward on title quality and humor.

## Setup

```
git clone https://github.com/asdafa3/abstract-to-title-generation
git submodule update --init
# pull the data stored in gdrive
dvc pull
```

