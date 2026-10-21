"""Dataset helpers for fine-tuning Mistral-7B to write multiple-choice questions.

Training and inference must use the same prompt, so both import
`build_prompt` from here.
"""

import json
from pathlib import Path

IGNORE_INDEX = -100  # label value the loss ignores

PROMPT_TEMPLATE = (
    "### Instruction:\n"
    "Generate a multiple choice question with 4 options and one correct answer "
    "based on the paragraph below.\n\n"
    "Paragraph: {paragraph}\n\n"
    "### Response:\n"
)


def build_prompt(paragraph: str) -> str:
    """Prompt for one paragraph, ending where the model's answer starts."""
    return PROMPT_TEMPLATE.format(paragraph=paragraph.strip())


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict]:
    """Read `{"instruction": paragraph, "response": mcq}` records, skipping blank lines."""
    records = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = {"instruction", "response"} - record.keys()
            if missing:
                raise ValueError(f"{path}:{line_no} is missing {sorted(missing)}")
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    return records


def tokenize_example(example: dict, tokenizer, max_length: int = 512) -> dict:
    """Tokenise one record for causal-LM fine-tuning.

    Only the response tokens (plus EOS, so the model learns to stop) count
    towards the loss; prompt tokens get label -100. No padding here: pad per
    batch with `DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100)`.
    """
    prompt = build_prompt(example["instruction"])
    answer = example["response"].strip() + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

    input_ids = (prompt_ids + answer_ids)[:max_length]
    labels = ([IGNORE_INDEX] * len(prompt_ids) + answer_ids)[:max_length]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}
