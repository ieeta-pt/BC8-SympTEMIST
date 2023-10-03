import pandas as pd
import click
from collections import defaultdict
from metrics import f1PR


@click.command()
@click.argument("run_file")
@click.argument("gs_file")
def main(run_file, gs_file):
    run_data = defaultdict(set)
    for i,entity in pd.read_csv(run_file,sep="\t").iterrows():
        run_data[entity["filename"]].add((entity["start_span"],entity["end_span"]))
        
    gs_data = defaultdict(set)
    for i,entity in pd.read_csv(gs_file,sep="\t").iterrows():
        gs_data[entity["filename"]].add((entity["start_span"],entity["end_span"]))
    
    run_data_filenames = set(run_data.keys())
    gs_data_filenames = set(gs_data.keys())
    #print(run_data_filenames)
    #print(gs_data_filenames)
    shared_filenames=run_data_filenames&gs_data_filenames

    print(len(gs_data_filenames), len(run_data_filenames), len(shared_filenames))
    
    if len(shared_filenames)!=run_data_filenames:
        print("Warning there are files wihtout prediciton")

    true_spans = []
    predicted_spans= []
    
    for k in shared_filenames:
        true_spans.append(gs_data[k])
        predicted_spans.append(run_data[k])
    
    tp = 0
    fn = 0
    fp = 0
    
    for i, j in zip(true_spans, predicted_spans):
        _tp = len(i.intersection(j))
        _fn = len(i.difference(j))
        _fp = len(j.difference(i))

        tp += _tp
        fn += _fn
        fp += _fp

    print(f1PR(tp, fn, fp))
    
if __name__ == '__main__':
    main()