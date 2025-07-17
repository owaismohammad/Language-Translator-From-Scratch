# ✨ Attention is All You Need – From Scratch + English to Kannada Translator (WIP)

Welcome to a personal deep dive into one of the most influential machine learning architectures of our time — the **Transformer**. This project is a full implementation of the seminal paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) — **from scratch** — using only Python and PyTorch.

Not just another copy-paste repo, this is a handcrafted, line-by-line reconstruction of the original transformer architecture, including:

- 🔸 Custom Multi-Head Attention  
- 🔸 Positional Encoding  
- 🔸 Layer Normalization  
- 🔸 Masking Strategies  
- 🔸 Feedforward Blocks  
- 🔸 Encoder and Decoder Stacks  
- 🔸 Training pipeline (data processing, batching)

All components have been built without relying on high-level libraries like `torch.nn.Transformer`. This was done to truly understand the **nuts and bolts** of how Transformers work under the hood.

---

## 🚀 Project Vision

I embarked on this project with one simple goal:

> **To implement a research paper from scratch and learn the architecture inside out.**

And what better candidate than the Transformer — the foundation for GPT, BERT, and every modern LLM today.

To apply the model in a meaningful way, I began building an **English to Kannada** translator using the architecture I implemented. This included writing:

- A custom tokenizer for both English and Kannada  
- Vocabulary builders  
- Data pipelines to prepare paired translations  
- Embedding layers integrated with positional encodings  
- A fully functional encoder-decoder setup with attention masking

---

## 🌐 English ➡️ Kannada Translator (Work-in-Progress)

Building a translator was the ultimate test of everything implemented — integrating embedding layers, positional information, masking logic, attention across encoder-decoder stacks, and more.

However, due to **hardware constraints** (limited GPU memory and CPU power), full-scale training wasn't feasible. As a result, while the architecture is entirely in place and trains on toy datasets, real-world translation is not fully realized **yet**.

Nonetheless, the translator pipeline is complete in design, and ready to be scaled once sufficient compute is available.

---

## 🧠 Challenges Faced

Re-implementing a 2017 research paper with no black-box utilities was a challenge in itself. Highlights include:

- Wrangling tensor dimensions across multi-head attention and masking  
- Ensuring causal decoding with future-masked self-attention  
- Writing layer normalization from scratch with custom epsilon tweaks  
- Debugging vanishing gradients and memory bottlenecks  
- Designing an interface where `Sequential` layers could handle nested modules like attention blocks and multi-step decoder logic

These struggles, though intense, led to immense growth in:
- Deep **PyTorch proficiency**  
- Understanding of **sequence-to-sequence** modeling  
- Respect for the genius of the original architecture

---

## 🛠️ Technologies Used

- 🐍 Python 3.10+  
- 🔥 PyTorch (Core APIs only — no high-level shortcuts!)  
- 🧾 JSONL + Custom Tokenizers  
- 📚 Research paper: [Attention is All You Need](https://arxiv.org/abs/1706.03762)

---

## 📁 Folder Structure

```bash
.
├── data/                  # Parallel corpora for English-Kannada (or toy data)
├── tokenizer/             # Custom tokenizer + vocabulary builders
├── transformer/           # Attention, encoder, decoder, mask, normalization
├── translator/            # Application logic for translation
├── train.py               # Training loop
├── utils.py               # Masking, batching, helpers
└── README.md              # You're here!
```

---

## 🌱 What's Next

- [ ] Improve batching and memory efficiency  
- [ ] Switch to mixed precision training (FP16)  
- [ ] Pretrain on larger English-Kannada datasets  
- [ ] Create a web interface for the translator  
- [ ] Optimize for CPU inference  

---

## 🧑‍💻 Final Words

This isn't just a machine learning project — it's a **journey into the heart of deep learning**, built one line of code at a time.

If you're someone who loves understanding things deeply, who wants to master PyTorch by implementing research ideas from scratch, or someone who just appreciates good engineering — I hope this project inspires you.

**Star** ⭐ the repo, fork it, study it, or reach out to collaborate.

Let’s build great things — one head at a time.

---

**Made with patience, PyTorch, and purpose.**
