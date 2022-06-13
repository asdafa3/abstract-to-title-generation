#!/bin/python
import os
import argparse
import textwrap
import pandas as pd
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Step through csv and write data for each cell.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input",
        metavar="INPUT",
        type=argparse.FileType('r', encoding="utf8"),
        help=".csv file to step through."
    )
    parser.add_argument(
        "-r",
        "--range",
        dest="range",
        metavar="R",
        help="Range of rows to step through.",
        type=int,
        nargs=2,
        default=None
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        default="output.csv",
        metavar="O",
        type=str,
        help=".csv file to output answers to."
    )
    return parser.parse_args()

def read_dataframe(args, skiprows, nrows):
    with args.input as input:
        #print(f"{skiprows} {nrows}")
        return pd.read_csv(input, skiprows=skiprows, nrows=nrows)

def format_string(s):
    return textwrap.indent('\n'.join(textwrap.wrap(s, 100)), '\t')

def print_row(id, title, abstract):
    fmt_title = format_string(f"\"{title}\"")
    print(f"ID {id}:\n\n{fmt_title}\n")
    print(f"\n{format_string(abstract)}\n")

def print_help():
    print("Allowed inputs:\n")
    for row in [
        ["[?, h]", "this help"],
        ["[0, 1, 2]", "a humor score for this title"],
        ["q", "quit"]
    ]:
        print(f"\t{row[0]:<10} - {row[1]:<10}")
    print('\n')

def ask_row_score():
    err = "Invalid score: Please input one of these numbers: [0, 1, 2]"
    while True:
        try:
            inp = input("How humorous is this title? [0, 1, 2], q, ?, h > ")
            match inp:
                case "?":
                    print_help()
                    continue
                case "q":
                    return None
                case _:
                    ()
            score = int(inp)
        except ValueError:
            print(err)
            continue
        if score not in range(0, 3):
            print(err)
            continue
        else:
            break
    #print(score)
    return score

def step_rows(df, nrows):
    scores=[]
    for index, row in df.iterrows():
        print(f"({index+1}/{nrows})", end=" ")
        id, title, abstract = row[0], row[1], row[2]
        print_row(id, title, abstract)
        score = ask_row_score()
        if score is not None:
            scores.append([id, score])
        else:
            break
        print('\n')
    return scores

def get_path_input():
    while True:
        try:
            val = input(f"Give file path to write to: ")
            if val == "":
                continue
            else:
                return Path(val)
        except ValueError:
            print("Invalid path")
            continue

def get_valid_path(outpath):
    while os.path.exists(outpath):
        ans = input(f"File {outpath} exists. Overwrite? (y/n) ")
        match ans:
            case "y":
                return outpath
            case "n":
                outpath = get_path_input()
            case _:
                continue
    return outpath
def write_scores(outpath, scores):
    path = get_valid_path(outpath)

def main(args):
    #print(args)
    if args.range:
        print("range")
        skiprows=args.range[0]
        nrows=args.range[1]-args.range[0]
    else:
        skiprows=None
        nrows=None
    df = read_dataframe(args, skiprows, nrows)
    nrows = len(df.index)
    scores = step_rows(df, nrows)
    outpath = get_valid_path(args.output)
    pd.DataFrame(scores, columns=['id', 'score']).to_csv(outpath, index=False)

if __name__ == "__main__":
    main(parse_args())