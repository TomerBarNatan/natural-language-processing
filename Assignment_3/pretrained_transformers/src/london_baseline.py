# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.
import argparse
import torch
import utils


if __name__ == "__main__":
    with open("birth_dev.tsv", mode="r", encoding="utf-8") as f:
        N = len(f.readlines())

    predictions = ["London"] * N
    total, correct = utils.evaluate_places(args.eval_corpus_path, predictions)
    if total > 0:
        print(
            "Correct: {} out of {}: {}%".format(correct, total, correct / total * 100)
        )
    else:
        print(
            "Predictions written to {}; no targets provided".format(args.outputs_path)
        )
