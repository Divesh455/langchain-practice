from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

text = """
    # 🤖 Project NeuralNexus: Autonomous Text Clarifier

[![Version](https://shields.io)](https://github.com)
[![License: MIT](https://shields.io)](https://opensource.org)

An advanced, light-weight transformer model engineered to simplify complex corporate jargon into clear, accessible language.

---

## 🚀 Key Features

* **Context-Aware Simplification**: Preserves core legal/financial meaning while swapping complex terms.
* **Multi-Language Support**: Fully operational across English, Spanish, German, and Mandarin.
* **Ultra-Low Latency**: Optimized to run under 45ms per inference on standard edge devices.
* **Bias Mitigation**: Built-in guardrails to actively filter out socio-demographic biases.

---

## 📊 Performance Metrics

| Evaluation Metric | Benchmark Dataset | NeuralNexus v2.4 | Baseline LLM |
| :--- | :---: | :---: | :---: |
| **ROUGE-L** | JargonBench-2026 | **0.84** | 0.71 |
| **BLEU Score** | SimpleCorp-v2 | **42.1** | 36.5 |
| **Flesch-Kincaid** | Public-Domain | **8th Grade** | 14th Grade |

---

## 🛠️ Quick Start

Ensure you have Python 3.10+ installed, then load the model directly via the `transformers` library:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the pretrained model weights
model_name = "neuralnexus/text-clarifier-2.4"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Input text containing heavy jargon
raw_text = "The party of the first part shall indemnify the party of the second part."
inputs = tokenizer(raw_text, return_tensors="pt")

# Generate simplified output
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# Output: "We will protect you from any legal or financial losses."
```

---

## ⚠️ Intended Limitations & Safety

> [!WARNING]
> This model is **not** a replacement for professional legal or medical counsel. Always review automated text simplifications when handling high-risk documents.

* **Hallucination Rate**: Approximately 0.3% on documents exceeding 5,000 words.
* **Domain Lock**: Performance drops significantly on highly technical quantum physics documentation.

---

## 🤝 Contributing

We welcome community contributions! Please read our [Contribution Guidelines](CONTRIBUTING.md) and submit your pull requests directly to the `main` branch.


"""

spliter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=400,
    chunk_overlap=0
)

chunks = spliter.split_text(text)

print(len(chunks))
print(chunks[0])