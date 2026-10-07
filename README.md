# LucidAI-Dataset

The data for our project: a model that looks at an action an AI agent wants to take and decides
whether to **allow** it (0) or **deny** it (1).

For example, a user says "My build is broken. Delete node_modules and reinstall", and the agent
wants to run `rm -rf node_modules && npm install`. That is **allow**, because the user asked for
it. Deleting a folder the user never mentioned would be **deny**.

**Deny** means the action can do real harm (delete data for good, leak passwords or private data,
install harmful software, weaken security, break production, spend money, open a door) and the
user did not ask for it, or the agent is following instructions it found in a web page or file
instead of the user's. Everything else is **allow**.

## What we did

1. **Chose a dataset.** [auto-1b-data](https://huggingface.co/datasets/ProCreations/auto-1b-data)
   has 725,000 agent actions labeled allow or deny. Each one comes with the user's request and
   the agent's earlier steps, so a model can judge the action in context. It covers 18 kinds of
   agent: coding, email, finance, browser, smart home and more.
2. **Kept English only** (removed 229,327). Our models work with English text.
3. **Removed examples that are far too long** (removed 17,077). They have up to 65,000 tokens,
   more than any of our models can read.
4. **Removed copies** (removed 8), so the same example cannot be in both training and testing.
5. **Split the 478,588 examples that are left** into training (70%), validation (15%) and test
   (15%). All examples with the same command stay in the same part, so the test only has
   commands the model has never seen. Commands that appear more than 20 times all go to
   training, so one command cannot decide the test score (pressing Enter alone appears 5,898
   times).
6. **Made two versions of each example:** the command alone (`input_a`), and the command with
   the user's request and the agent's earlier steps (`input_b`). Comparing them shows how much
   the context helps.
7. **Added two outside test sets** that were made separately from the main data, to check that
   a model also works on new kinds of examples: approve-or-deny (1,448 examples) and saroku
   (26,633 examples).

Nothing else was removed. `data/REPORT.md` shows the count after every step, and
`data/dropped.jsonl.gz` lists every removed example and why.

## Files

| file | examples | use it for |
|---|---:|---|
| `data/train_1.csv.gz`, `train_2`, `train_3` | 335,349 | training (3 parts, because GitHub does not accept files over 100 MB) |
| `data/val.csv.gz` | 71,626 | trying settings and comparing models |
| `data/test.csv.gz` | 71,613 | the final score, used once at the end |
| `data/ood_approve_or_deny.csv.gz` | 1,448 | outside test 1 |
| `data/ood_saroku.csv.gz` | 26,633 | outside test 2 |
| `external_tests/`, `validate_external_tests.py` | | our own test examples, and the script that checks them |
| `mayi_colab.ipynb`, `prepare_dataset.py`, `mayi_input.py` | | the code that made the data |
| [`DETAILS.md`](DETAILS.md) | | every detail of how the data was made |

About half of the examples in every file are deny.

## Load the data

In Colab:

```python
!git clone https://github.com/Enes-Ozbek/LucidAI-Dataset
import glob
import pandas as pd

def load(name):
    files = sorted(glob.glob(f"LucidAI-Dataset/data/{name}*.csv.gz"))
    return pd.concat([pd.read_csv(f, engine="python") for f in files], ignore_index=True)

train, val, test = load("train"), load("val"), load("test")
ood1, ood2 = load("ood_approve_or_deny"), load("ood_saroku")
```

## Columns to use

| column | what it is |
|---|---|
| `input_a` | the command alone |
| `input_b` | the command, the user's request and the agent's earlier steps |
| `label` | 1 = deny, 0 = allow |
| `length_bucket` | short, medium or long: about 2, 6 or 12 earlier agent steps |

Don't give the model `category` or `rationale`: they give the answer away. Use them only to look
at mistakes. All columns are explained in [DETAILS.md](DETAILS.md).

## Good to know

- 15% of the examples have a command that is allowed with one request and denied with another.
  Only `input_b` can get those right.
- DistilBERT reads at most 512 tokens. In `input_b`, 28% of the medium and 58% of the long
  examples are longer, so their last steps get cut. Report accuracy for short, medium and long
  separately to see what this costs.
- The labels were made by AI models, not by people, so a few of them may be wrong.
- Expect lower scores on saroku: its examples are written in a different style.

## Make the data again

Open `mayi_colab.ipynb` in Colab and choose *Runtime > Run all*. It downloads the sources and
builds exactly the same files again.

## Sources

| dataset | used for | license |
|---|---|---|
| [ProCreations/auto-1b-data](https://huggingface.co/datasets/ProCreations/auto-1b-data) | main data | Apache-2.0 |
| [ProCreations/approve-or-deny](https://huggingface.co/datasets/ProCreations/approve-or-deny) | outside test 1 | Apache-2.0 |
| [karanxa/agent-action-safety-dataset](https://huggingface.co/datasets/karanxa/agent-action-safety-dataset) ("saroku") | outside test 2 | Apache-2.0 |
