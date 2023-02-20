import argparse
import json
import os
PROJECT_ROOT = os.path.dirname(__file__)
# DEFAULT_MODEL='/model'
DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "models", "deberta")

def parse_args():
    parser = argparse.ArgumentParser(description='This is a baseline for task 2 that spoils each clickbait post with the title of the linked page.')

    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The spoiled posts in jsonl format.', required=False)
    parser.add_argument('--model', type=str, help='The path to the model.', default=DEFAULT_MODEL)

    return parser.parse_args()
args = parse_args()