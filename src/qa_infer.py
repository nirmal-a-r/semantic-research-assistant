import torch


def qa_predict_single(model, tokenizer, question, context, max_length=384, device="cpu"):
    """Extractive QA inference. Full implementation in Section 17 of the notebook."""
    inputs = tokenizer(
        question, context, return_tensors="pt", truncation="only_second",
        max_length=max_length, return_offsets_mapping=True,
    ).to(device)
    offset_mapping = inputs.pop("offset_mapping")[0].tolist()
    with torch.no_grad():
        outputs = model(**inputs)
    start = int(torch.argmax(outputs.start_logits))
    end = int(torch.argmax(outputs.end_logits))
    if end < start or offset_mapping[start] is None or offset_mapping[end] is None:
        return ""
    s, e = offset_mapping[start][0], offset_mapping[end][1]
    return context[s:e]
