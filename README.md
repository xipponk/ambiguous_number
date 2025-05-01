# 🔢 Ambiguous Number Generator 🇹🇭 — Where Hope Meets Hallucination

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Made with ❤️](https://img.shields.io/badge/Made%20with-%E2%9D%A4-red)

> A fun cultural-AI project using Python + OpenCV to generate Thai-style ambiguous digits — inspired by traditional lottery calendars. Perfect for generative art, satire, or human-computer ambiguity studies.

---

## 🧠 What Is This?

This project generates synthetic images of ambiguous Thai digits — digits that look like 7 (or maybe 1?), drawn with distortion, blur, and layered randomness. The visuals are inspired by hand-written numbers found in Thai traditional lottery calendars.

---

## ✨ Features

- Generate random hand-written digits with distortion
- Gaussian blur, motion blur, and noise injection
- Multiple digits layered for higher ambiguity
- Thai font support (just drop your `.ttf` files into `fonts/`)
- Easy to extend for generative AI experiments

---

## 📦 Installation

```bash
git clone https://github.com/xipponk/ambiguous_number.git
cd ambiguous_number
python -m venv .venv-ambiguous-ai
.venv-ambiguous-ai\Scripts\activate
pip install -r requirements.txt

🚀 Usage
Run the script to generate 100 ambiguous digit images:

bash
Copy
Edit
python src/generate_effects.py
Images will be saved into:

Copy
Edit
ambiguous_digits_opencv/
📁 Folder Structure
bash
Copy
Edit
ambiguous_number/
├── src/
│   └── generate_effects.py       # Main generator script using OpenCV + PIL
├── fonts/                        # Add custom Thai-style handwriting fonts here
├── ambiguous_digits_opencv/     # Output images go here
├── requirements.txt
└── README.md
🎯 Use Cases
Cultural tech and design research

Ambiguity in human-digit recognition

Satirical experiments in lottery prediction

Generative AI fine-tuning

Game/UX randomness exploration

📜 License
This project is licensed under the MIT License — feel free to remix, expand, or spiritualize responsibly.

⚠️ No lottery wins guaranteed. AI only generates hopes, not results.