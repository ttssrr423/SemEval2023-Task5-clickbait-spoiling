import os
import random
import json
script_path = os.path.dirname(__file__)

def start_split():
    original_val_path = os.path.join(script_path, "..", "dataset", "validation.jsonl.bkup")
    new_val_path = os.path.join(script_path, "..", "dataset", "validation.jsonl")
    new_test_path = os.path.join(script_path, "..", "dataset", "test.jsonl")
    print(original_val_path)
    # val_inp = [json.loads(i) for i in open(original_val_path, "r", encoding="utf8")]
    val_inp = [i for i in open(original_val_path, "r", encoding="utf8")]
    random.shuffle(val_inp)
    test_num = len(val_inp)//2

    new_vals = val_inp[:test_num]
    new_tests = val_inp[test_num:]
    print(len(val_inp), test_num)

    fw = open(new_val_path, mode="w", encoding="utf8")
    fw.write("".join(new_vals))
    fw.close()

    fw2 = open(new_test_path, mode="w", encoding="utf8")
    fw2.write("".join(new_tests))
    fw2.close()
    print("split done...")

def analyse_cts():
    new_val_path = os.path.join(script_path, "..", "dataset", "validation.jsonl")
    new_test_path = os.path.join(script_path, "..", "dataset", "test.jsonl")
    val_entities = [json.loads(i) for i in open(new_val_path, "r", encoding="utf8")]
    test_entities = [json.loads(i) for i in open(new_test_path, "r", encoding="utf8")]

    from collections import Counter
    val_stat = Counter()
    for item in val_entities: val_stat[item["tags"][0]] += 1

    test_stat = Counter()
    for item in test_entities: test_stat[item["tags"][0]] += 1

    print("val", val_stat)
    print("test", test_stat)

if __name__ == "__main__":
    # start_split()
    analyse_cts()