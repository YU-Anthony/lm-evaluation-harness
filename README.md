# Language Model Evaluation Harness

![](https://github.com/EleutherAI/lm-evaluation-harness/workflows/Build/badge.svg)
[![codecov](https://codecov.io/gh/EleutherAI/lm-evaluation-harness/branch/master/graph/badge.svg?token=JSG3O2427J)](https://codecov.io/gh/EleutherAI/lm-evaluation-harness)

## Overview 

This project initially comes from this [resporitory](https://github.com/EleutherAI/lm-evaluation-harness) and provides a unified framework to test autoregressive language models (GPT-2, GPT-3, GPTNeo, etc) on a large number of different evaluation tasks.

I added several changes so that it can supprt T5 realted models

Features:

- 200+ tasks implemented
- Support for GPT-2, GPT-3, GPT-Neo, GPT-NeoX, and GPT-J, T5, T0 with flexible tokenization-agnostic interface
- Task versioning to ensure reproducibility

## Install

```bash
pip install lm-eval
```

## Basic Usage

To evaluate a model, (e.g. GPT-2) on NLU tasks (e.g. LAMBADA, HellaSwag), you can run the following command. **When reporting results from eval harness, please include the task versions (shown in `results["versions"]`) for reproducibility.** This allows bug fixes to tasks while also ensuring that previously reported scores are reproducible. See the [Task Versioning](https://github.com/EleutherAI/lm-evaluation-harness#task-versioning) section for more info.

```bash
python main.py \
	--model gpt2 \
	--device cuda:0 \
	--tasks lambada,hellaswag
```
(This uses gpt2-117M by default as per HF defaults, use --model_args to specify other gpt2 sizes)

Additional arguments can be provided to the model constructor using the `--model_args` flag. Most importantly, the `gpt2` model can be used to load an arbitrary HuggingFace model. For example, to run GPTNeo use the following:

```bash
python main.py \
	--model gpt2 \
	--model_args pretrained=EleutherAI/gpt-neo-2.7B \
	--device cuda:0 \
	--tasks lambada,hellaswag
```

If you have access to the OpenAI API, you can also evaluate GPT-3:

```bash
export OPENAI_API_SECRET_KEY=YOUR_KEY_HERE
python main.py \
	--model gpt3 \
	--model_args engine=davinci \
	--tasks lambada,hellaswag
```

To evaluate T5:
```bash
python main.py \
  --model t5  \
  --model_args pretrained=google/t5-large-lm-adapt  \
  --device cuda:0  \
  --tasks lambada

```

You can also evaluate T0 realted models:
```bash
python main.py \
  --model t0  \
  --model_args pretrained=bigscience/T0_3B  \
  --device cuda:0  \
  --tasks lambada

```

And if you want to verify the data integrity of the tasks you're performing in addition to running the tasks themselves, you can use the `--check_integrity` flag:

```bash
python main.py \
	--model gpt3 \
	--model_args engine=davinci \
	--tasks lambada,hellaswag \
	--check_integrity
```
To evaluate mesh-transformer-jax models that are not available on HF, please invoke eval harness through [this script](https://github.com/kingoflolz/mesh-transformer-jax/blob/master/eval_harness.py).

## Implementing new tasks

To implement a new task in eval harness, see [this guide](./docs/task_guide.md).

## Results
Here are some results for LAMBADA task:
|Task|Model|Metric|Value||Stderr|
|-------|------:|------|-------:|---|------:|
|lambada|        gpt2      |ppl| 40.0554|±| 1.4881 |
|       |                  |acc| 0.3256 |±| 0.0065 |
|       |T5-small-lm-adapt |ppl|777.2453|±|102.2926|
|       |                  |acc| 0.6053 |±| 0.0068 |
|       | T5-base-lm-adapt |ppl|4285.7492|±|610.3834|
|       |                  |acc| 0.3996 |±| 0.0068 |
|       |T5-large-lm-adapt |ppl|1576.739|±|204.8895|
|       |                  |acc| 0.418 |±| 0.0069 |
|       |     T0_3B        |ppl|113079.6175|±|17162.7004|
|       |                  |acc| 0.0470 |±| 0.0029 |
|       |     T0pp        |ppl|6552.9149|±|757.4522|
|       |                  |acc| 0.0745 |±| 0.0037 |

Here are results from LAMBADA paper:
![](https://img-blog.csdnimg.cn/f170ef21774d4998aa8063657d273c7c.png?x-oss-process=image)

## Cite as

```
@software{eval-harness,
  author       = {Gao, Leo and
                  Tow, Jonathan and
                  Biderman, Stella and
                  Black, Sid and
                  DiPofi, Anthony and
                  Foster, Charles and
                  Golding, Laurence and
                  Hsu, Jeffrey and
                  McDonell, Kyle and
                  Muennighoff, Niklas and
                  Phang, Jason and
                  Reynolds, Laria and
                  Tang, Eric and
                  Thite, Anish and
                  Wang, Ben and
                  Wang, Kevin and
                  Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = sep,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.0.1},
  doi          = {10.5281/zenodo.5371628},
  url          = {https://doi.org/10.5281/zenodo.5371628}
}
```

