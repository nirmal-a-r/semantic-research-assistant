import evaluate as hf_evaluate

squad_metric = hf_evaluate.load("squad")


def compute_squad_metrics(predictions, references):
    """predictions: list[{"id": ..., "prediction_text": ...}]
    references:   list[{"id": ..., "answers": ...}]
    """
    return squad_metric.compute(predictions=predictions, references=references)
