# TODO: [part d]
# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import argparse
import utils

def main():
    accuracy = 0.0

    # Compute accuracy in the range [0.0, 100.0]
    ### YOUR CODE HERE ###
    with open("birth_dev.tsv", 'r') as file:
        line_count = sum(1 for line in file)
    predictions = ["London"] * line_count
    total, correct = utils.evaluate_places("birth_dev.tsv", predictions)
    accuracy = correct/total * 100
    ### END YOUR CODE ###

    return accuracy

if __name__ == '__main__':
    accuracy = main()
    with open("london_baseline_accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"{accuracy}\n")
