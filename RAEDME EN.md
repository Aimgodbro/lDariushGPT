# DariushGPT - Persian Multitask Language Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Transformer-based model for Persian NLP supporting text generation, poetry generation, and sentiment analysis.

## Features
- Coherent text generation
- Poetry generation with prosodic constraints
- Sentiment analysis (93% accuracy)
- MPS, CUDA, and CPU support

## Installation
```bash
git clone https://github.com/Aimgodbro/DariushGPT.git
cd DariushGPT
pip install -r requirements.txt
```

## Usage
```python
from src.model.architecture import DariushGPT

model = DariushGPT()
poem = model.generate_poem(bahr="hazaj")
print(poem)
```

## Contributing
Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details.
