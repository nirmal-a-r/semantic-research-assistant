from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

ROOT = Path(__file__).resolve().parent.parent


def _flatten_squad(articles):
    rows = []
    for article in articles:
        for para in article["paragraphs"]:
            ctx = para["context"]
            for qa in para["qas"]:
                rows.append({
                    "id":       qa["id"],
                    "title":    article.get("title", ""),
                    "context":  ctx,
                    "question": qa["question"],
                    "answers":  {
                        "text":         [a["text"]         for a in qa["answers"]],
                        "answer_start": [a["answer_start"] for a in qa["answers"]],
                    },
                })
    return rows


def load_squad(train_subset=None, eval_subset=None, seed=42, data_dir=None):
    """Load SQuAD v1.1 from the local data/raw/*.json files (already
    downloaded — see data/raw/train.json, data/raw/dev.json) instead of
    hitting the Hugging Face Hub. Mirrors scripts/train_full.py's loader.
    """
    data_dir = Path(data_dir) if data_dir else (ROOT / "data")
    local_train = str(data_dir / "raw" / "train.json")
    local_dev   = str(data_dir / "raw" / "dev.json")

    squad_json = load_dataset("json",
                               data_files={"train": local_train, "validation": local_dev},
                               field="data")
    raw = DatasetDict({
        "train":      Dataset.from_list(_flatten_squad(squad_json["train"])),
        "validation": Dataset.from_list(_flatten_squad(squad_json["validation"])),
    })

    train = (raw["train"].shuffle(seed=seed).select(range(train_subset))
             if train_subset else raw["train"])
    val = (raw["validation"].shuffle(seed=seed).select(range(eval_subset))
           if eval_subset else raw["validation"])
    return train, val


def build_corpus(*splits):
    seen, corpus = {}, []
    for split in splits:
        for ctx, title in zip(split["context"], split["title"]):
            if ctx not in seen:
                doc_id = f"doc_{len(corpus)}"
                seen[ctx] = doc_id
                corpus.append({"doc_id": doc_id, "title": title, "text": ctx})
    return corpus, seen
