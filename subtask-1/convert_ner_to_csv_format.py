import json
import click
import pandas as pd
import os

@click.command()
@click.argument("run_json")
@click.option("--output_folder", default="runs_tsv_format")
def main(run_json, output_folder):
    
    with open(run_json) as f:
        run_data = json.load(f)
        
    column_name = ["filename", "label", "start_span", "end_span", "text"]
    
    clean_data = []
    for doc_id, data in run_data.items():
        entities = data["decoder"]
        assert len(entities["span"]) == len(entities["text"])
        for i in range(len(entities["span"])):
            clean_data.append([doc_id,"SINTOMA",entities["span"][i][0], entities["span"][i][1], entities["text"][i]])

    df = pd.DataFrame(clean_data,columns=column_name)
    
    basename,_ = os.path.splitext(os.path.basename(run_json))
    out_name = os.path.join(output_folder, f"{basename}.tsv")
    df.to_csv(out_name, sep="\t", index = False)

if __name__ == '__main__':
    main()
