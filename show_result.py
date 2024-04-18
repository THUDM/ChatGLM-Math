"""
Usage:
python3 show_result.py --mode [single|pairwise-baseline|pairwise-all]
"""
import argparse
import pandas as pd
import jsonlines
import numpy as np
import os


def load_categories(filename):
    qids = {}
    with jsonlines.open(filename, "r") as f:
        for doc in f:
            qids[doc["question_id"]] = doc["category"]
        f.close()
    return qids

def load_second_categories(filename):
    qids = {}
    with jsonlines.open(filename, "r") as f:
        for doc in f:
            qids[doc["question_id"]] = doc["category"] + "-" + doc["subcategory"]
        f.close()
    return qids

def main(args):
    if args.input_dir is None:
        input_dir = (
            "data/judgment"
        )
    else:
        input_dir = args.input_dir
    
    input_files = os.listdir(input_dir)
    input_files = [os.path.join(input_dir, file) for file in input_files if file.endswith(".jsonl")]

    print(f"{len(input_files)} Input files: ", input_files)
    for input_file in input_files:
        print(input_file)
        with jsonlines.open(input_file, "r") as f:
            for doc in f:
                pass
        pd.read_json(input_file, lines=True)
    df_all = pd.concat([pd.read_json(input_file, lines=True) for input_file in input_files])

    # create file
    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)

    categories = load_categories(args.ques_file)
    second_categories = load_second_categories(args.ques_file)

    math_key = "math-user-eval"
    splits = [math_key,f"{math_key}-hard-split"]
    subsplits = list(set(second_categories.values()))
    splits.extend(subsplits)
    categories_list = list(set(categories.values()))
    splits.extend(categories_list)
    
    print("> loaded split: ", splits)
    total = {}
    idx = 0
    for _, line in df_all.iterrows():
        if line["question_id"] not in categories:
            idx = idx + 1
            continue
        cat = categories[line["question_id"]]
        sub_cat = second_categories[line["question_id"]]
        model = line["model_id"]
        if model not in total:
            total[model] = {split:[] for split in splits}
            total[model]["TOTAL"] = []
        total[model][cat].append(line["score"])
        total[model][sub_cat].append(line["score"])
        total[model][math_key].append(line["score"])
    class_df = []
    print(idx)
    for model in total.keys():  
        model_info = [model]
        model_info.append(np.mean(total[model][math_key]))
        model_info.append(len([i for i in total[model][math_key] if i >= 7])/len(total[model][math_key]))
        for split in splits[2:]:
            model_info.append(np.mean(total[model][split]))
        class_df.append(model_info)
    
    columns = ["model"] + splits
    class_df = pd.DataFrame(class_df, columns=columns)
    class_df = class_df.sort_values(by=math_key, ascending=False)
    print("\n########## MathUserEval Results ##########")
    print(class_df)
    class_df.to_excel(args.save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-file", type=str, default="data/results/results.xlsx")
    parser.add_argument("--input-dir", type=str)
    parser.add_argument("--ques-file", type=str)
    args = parser.parse_args()

    main(args)
