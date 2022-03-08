"""
Language Models are Few-Shot Learners
https://arxiv.org/pdf/2005.14165.pdf

Unscramble is a small battery of 5 “character manipulation” tasks. Each task
involves giving the model a word distorted by some combination of scrambling,
addition, or deletion of characters, and asking it to recover the original word.

Homepage: https://github.com/openai/gpt-3/tree/master/data
"""
import gzip
import json
import shutil
from pathlib import Path
from best_download import download_file
from lm_eval.base import Task, rf
from lm_eval.metrics import mean


_CITATION = """
@inproceedings{NEURIPS2020_1457c0d6,
    author = {Brown, Tom and Mann, Benjamin and Ryder, Nick and Subbiah, Melanie and Kaplan, Jared D and Dhariwal, Prafulla and Neelakantan, Arvind and Shyam, Pranav and Sastry, Girish and Askell, Amanda and Agarwal, Sandhini and Herbert-Voss, Ariel and Krueger, Gretchen and Henighan, Tom and Child, Rewon and Ramesh, Aditya and Ziegler, Daniel and Wu, Jeffrey and Winter, Clemens and Hesse, Chris and Chen, Mark and Sigler, Eric and Litwin, Mateusz and Gray, Scott and Chess, Benjamin and Clark, Jack and Berner, Christopher and McCandlish, Sam and Radford, Alec and Sutskever, Ilya and Amodei, Dario},
    booktitle = {Advances in Neural Information Processing Systems},
    editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
    pages = {1877--1901},
    publisher = {Curran Associates, Inc.},
    title = {Language Models are Few-Shot Learners},
    url = {https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf},
    volume = {33},
    year = {2020}
}
"""


def extract_gzip(gz, to):
    with gzip.open(gz, 'rb') as fin:
        with open(to, 'wb') as fout:
            shutil.copyfileobj(fin, fout)


class WordUnscrambleTask(Task):
    VERSION = 0
    BASE_PATH = Path("data/unscramble")
    FILENAME = None
    CHECKSUM = None  # SHA256 Checksum.

    def __init__(self):
        super().__init__()

    def download(self):
        if not self.BASE_PATH.exists():
            Path.mkdir(self.BASE_PATH, parents=True)
        file = self.BASE_PATH / self.FILENAME
        if not file.exists():
            rawfile = file.parent / (file.name + ".gz")
            base_url = "https://raw.githubusercontent.com/openai/gpt-3/master/data"
            download_file(f"{base_url}/{self.FILENAME}.gz", local_file=str(rawfile), expected_checksum=self.CHECKSUM)
            extract_gzip(gz=rawfile, to=file)

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        file = self.BASE_PATH / self.FILENAME
        return (json.loads(line) for line in open(file).read().splitlines())

    def doc_to_text(self, doc):
        return doc["context"]

    def doc_to_target(self, doc):
        return doc["completion"]

    def construct_requests(self, doc, ctx):
        completion = rf.greedy_until(ctx, ["\n"])
        return completion

    def process_results(self, doc, results):
        pred = results[0]
        gold = doc["completion"]
        return {
            "acc": int(pred == gold)
        }

    def aggregation(self):
        return {
            "acc": mean
        }

    def higher_is_better(self):
        return {
            "acc": True
        }


class Anagrams1(WordUnscrambleTask):
    FILENAME = "mid_word_1_anagrams.jsonl"
    CHECKSUM = "6768a86896083199de4815d4964cb2f6f1046476cfd80c2a562784f182905979"


class Anagrams2(WordUnscrambleTask):
    FILENAME = "mid_word_2_anagrams.jsonl"
    CHECKSUM = "c3d839d09a7954b78a27cd2cd75d4ed0488656c56ef4dbd741a005343826cb01"


class CycleLetters(WordUnscrambleTask):
    FILENAME = "cycle_letters_in_word.jsonl"
    CHECKSUM = "1689c9002bb8c5988bf5f05e977c9db92f57932c1b5a38998c29ac0dd71e1d42"


class RandomInsertion(WordUnscrambleTask):
    FILENAME = "random_insertion_in_word.jsonl"
    CHECKSUM = "72e65d83da53d15752ee0c47379509de149ddbad32d61184e5991df29616b78a"


class ReversedWords(WordUnscrambleTask):
    FILENAME = "reversed_words.jsonl"
    CHECKSUM = "133a08f875cd6c1ef8608a3233571a773881cc27b1c707de738cc6543439332a"