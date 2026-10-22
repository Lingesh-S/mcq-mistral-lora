# 🧠 MCQ Generator using Mistral-7B + LoRA (Fine-tuned)

This repository contains training and inference notebooks for fine-tuning the Mistral 7B model using the [LoRA](https://arxiv.org/abs/2106.09685) technique to generate Multiple Choice Questions (MCQs) from academic paragraphs.

---

## 📦 Files Included

```
mcq-mistral-lora/
├── mcq_mistral_lora_ft.ipynb   # QLoRA fine-tuning, evaluation, push to Hub, inference
├── mcq_lora/data.py            # Prompt template + preprocessing shared by training and inference
├── data/sample.jsonl           # 5 example records showing the dataset format
├── tests/                      # CPU-only tests for the data pipeline
├── requirements.txt            # GPU training / inference dependencies
└── requirements-dev.txt        # pytest, ruff, nbformat (no GPU needed)
```

---

## 🔗 Model on Hugging Face

👉 [Lingesh-S/mcq-mistral-lora](https://huggingface.co/Lingesh-S/mcq-mistral-lora)

---

## 🚀 How to Use

### 🧪 Fine-tuning

Open [`mcq_mistral_lora_ft.ipynb`](mcq_mistral_lora_ft.ipynb) in Google Colab with a **T4 GPU** and run all cells. It:

* loads Mistral-7B-Instruct in **4-bit NF4** (QLoRA) with `bitsandbytes`
* attaches LoRA adapters (`r=8`, `lora_alpha=32`, `dropout=0.05`, `q_proj`/`v_proj`) with `peft`
* masks the prompt so only the answer tokens are trained on, and appends EOS so the model learns to stop
* trains with the Hugging Face `Trainer` (3 epochs, effective batch 8, cosine LR 2e-4) and reports eval loss and perplexity
* saves the adapter and can push it to the Hub

Put your dataset at `data/<name>.jsonl` (git-ignored) and set `DATA_PATH`. Each line is
`{"instruction": "<paragraph>", "response": "<question, 4 options, correct answer>"}`; see `data/sample.jsonl`.

### 📘 Inference

The last section of the notebook loads the base model plus the adapter and calls `generate_mcq(paragraph)`.
It uses `mcq_lora.build_prompt`, the same template as training.

### 🧪 Checks without a GPU

```bash
pip install -r requirements-dev.txt
ruff check . && pytest
```

---

## 🧾 Requirements

```bash
pip install -r requirements.txt
```

---

## 📊 Training Summary

* **Model**: `mistralai/Mistral-7B-Instruct-v0.1`
* **Adapter**: LoRA (`r=8`, `lora_alpha=32`, `dropout=0.05`, `q_proj`, `v_proj`)
* **Training samples**: \~500 JSONL examples (private dataset)
* **Epochs**: 3
* **Final loss**: \~0.23 (from the original Colab run; the notebook now prints train and eval loss so a re-run can confirm it)

---

## ✍️ Example Prompt

```text
### Instruction:
Generate a multiple choice question with 4 options and one correct answer based on the paragraph below.

Paragraph: The Lok Sabha is the House of the People in India. It is one of the two houses of Parliament.
```

### ✅ Output

```text
What is the name of the House of the People in India?

a) The Rajya Sabha
b) The Lok Sabha
c) The Supreme Court
d) The President's House

Correct answer: b) The Lok Sabha
```

---

## 📩 Author

This project is created and maintained by **Lingesh S**.
Feel free to connect via [Hugging Face](https://huggingface.co/Lingesh-S) or [LinkedIn](https://www.linkedin.com/in/lingesh-s29).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
