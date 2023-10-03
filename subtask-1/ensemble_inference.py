import json
import click
from ensemble import ensemble_entity_level

@click.command()
@click.argument('runs', nargs=-1, type=click.Path())
@click.option('--out')
def main(runs, out):
    
    model_preds = []
    
    
    
    for run in runs:
        with open(run) as f:
            model_preds.append(json.load(f))

    doc_ids = list(model_preds[0].keys())
    
    ensemble_output = {}
    for doc_id in doc_ids:
        document_text =  model_preds[0][doc_id]["document_text"]
        ensemble_output[doc_id]={"decoder": ensemble_entity_level([model_preds[i][doc_id]["decoder"] for i in range(len(model_preds))], document_text),
                                 "document_text": document_text}
        
    with open(out, "w") as fOut:
        fOut.write(json.dumps(ensemble_output))
    
if __name__ == '__main__':
    main()