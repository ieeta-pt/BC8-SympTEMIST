from decoder import decoder
from trainer import NERevalPrediction


def f1PR(tp, fn, fp):
    precision = 0 if tp == 0 else tp / (tp + fp)
    recall = 0 if tp == 0 else tp / (tp + fn)
    f1 = 0 if precision == 0 and recall == 0 else (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


class NERMetrics:

    def __init__(self, context_size=64):
        self.context_size = context_size

    def __call__(self, evaluationOutput: NERevalPrediction):
        # evaluationOutput.metadata = {"doc_id", "sequence_id", "offsets", "og_annotations"}
        # evaluationOutput.predictions
        doc_gs = {}
        documents = {}
        padding = self.context_size

        # reconsturct the document in the correct order
        for i in range(len(evaluationOutput.metadata)):
            doc_id = evaluationOutput.metadata[i]['doc_id']
            if doc_id not in documents.keys():
                documents[doc_id] = {}
                doc_gs[doc_id] = [(ann['start_span'], ann['end_span']) for ann in
                                  evaluationOutput.metadata[i]["og_annotations"]]
                # run 1 time is enough for this stuff

            documents[doc_id][evaluationOutput.metadata[i]['sequence_id']] = {
                'labels': evaluationOutput.predictions[i].tolist(),
                'offsets': evaluationOutput.metadata[i]['offsets']}

        print("DOCUMENTS:", len(documents))

        predicted_spans = []
        true_spans = []
        # decode each set of labels and store the offsets
        for doc in documents.keys():
            current_doc = [documents[doc][seq]['labels'] for seq in sorted(documents[doc].keys())]
            current_offsets = [documents[doc][seq]['offsets'] for seq in sorted(documents[doc].keys())]
            predicted_spans.append(decoder(current_doc, current_offsets, padding=padding)["span"])
            true_spans.append(doc_gs[doc])

        # compute f1 score
        tp = fn = fp = 0

        assert len(true_spans) == len(predicted_spans)
        samples_f1 = 0
        samples_p = 0
        samples_r = 0

        for i, j in zip(true_spans, predicted_spans):
            _tp = len(set(i).intersection(set(j)))
            _fn = len(set(i).difference(set(j)))
            _fp = len(set(j).difference(set(i)))

            s_f1, s_p, s_r = f1PR(_tp, _fn, _fp)
            samples_f1 += s_f1
            samples_p += s_p
            samples_r += s_r

            tp += _tp
            fn += _fn
            fp += _fp

        micro_f1, micro_p, micro_r = f1PR(tp, fn, fp)

        return {"microF1": micro_f1,
                "microP": micro_p,
                "microR": micro_r,
                "macroF1": samples_f1 / len(true_spans),
                "macroP": samples_p / len(true_spans),
                "macroR": samples_r / len(true_spans)}
