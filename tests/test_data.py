import json
from pathlib import Path

import pytest

from mcq_lora.data import IGNORE_INDEX, build_prompt, load_jsonl, tokenize_example

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample.jsonl"


class CharTokenizer:
    """Minimal tokenizer: one id per character, BOS = 1, EOS = 2."""

    eos_token = "</s>"

    def __call__(self, text, add_special_tokens=True):
        ids = []
        while text:
            if text.startswith(self.eos_token):
                ids.append(2)
                text = text[len(self.eos_token):]
            else:
                ids.append(ord(text[0]) + 10)
                text = text[1:]
        return {"input_ids": ([1] if add_special_tokens else []) + ids}


def test_prompt_matches_documented_format():
    prompt = build_prompt("  Some paragraph.  ")
    assert prompt.startswith("### Instruction:\n")
    assert "Paragraph: Some paragraph.\n" in prompt
    assert prompt.endswith("### Response:\n")


def test_sample_dataset_is_valid():
    records = load_jsonl(SAMPLE)
    assert len(records) == 5
    assert all("Correct answer:" in r["response"] for r in records)


def test_load_jsonl_limit_and_validation(tmp_path):
    good = tmp_path / "good.jsonl"
    good.write_text("\n".join(json.dumps({"instruction": f"p{i}", "response": "r"}) for i in range(3)) + "\n\n")
    assert len(load_jsonl(good, limit=2)) == 2

    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"instruction": "p"}) + "\n")
    with pytest.raises(ValueError, match="response"):
        load_jsonl(bad)


def test_only_answer_tokens_are_trained_on():
    tok = CharTokenizer()
    example = {"instruction": "Para.", "response": "Q?"}
    out = tokenize_example(example, tok)
    prompt_len = len(tok(build_prompt("Para."))["input_ids"])

    assert len(out["input_ids"]) == len(out["labels"]) == len(out["attention_mask"])
    assert out["labels"][:prompt_len] == [IGNORE_INDEX] * prompt_len
    assert out["labels"][prompt_len:] == out["input_ids"][prompt_len:]
    assert out["input_ids"][-1] == 2  # ends with EOS so the model learns to stop


def test_truncation_keeps_lengths_aligned():
    out = tokenize_example({"instruction": "x" * 100, "response": "y" * 100}, CharTokenizer(), max_length=50)
    assert len(out["input_ids"]) == len(out["labels"]) == 50
