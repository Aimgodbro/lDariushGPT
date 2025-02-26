### **README.md**  

---

# **DariushGPT**  
**A Versatile AI Model for Persian Language Processing**  
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)  
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)  
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)  

---

## **Introduction**  
**DariushGPT** is an advanced AI model for Persian language processing, built on the **Transformer** architecture. This model offers various capabilities, including text generation, sentiment analysis, poetry composition, chain-of-thought reasoning, and automatic translation. By leveraging cutting-edge techniques such as **Mixture of Experts (MoE)**, **Rotary Positional Embeddings (RoPE)**, and **Retrieval-Augmented Generation (RAG)**, DariushGPT has become a powerful tool for multiple applications.  

---

## **Key Features**  
✅ **Versatile Capabilities:**  
   - Text generation, poetry, sentiment analysis, automatic translation, and chain-of-thought reasoning.  

✅ **Advanced Architecture:**  
   - Uses **Transformer** with **FlashAttention** and **RoPE**.  
   - Implements **Mixture of Experts (MoE)** for specialized processing.  

✅ **Reinforcement Learning:**  
   - Supports **Reinforcement Learning from Human Feedback (RLHF)** and **Direct Preference Optimization (DPO)**.  

✅ **Multimodal Processing:**  
   - Integration with **CLIP** (image processing) and **ASR** (audio processing).  

✅ **RAG System:**  
   - Accesses external knowledge using **FAISS** for information retrieval.  

✅ **Intelligent Text Generation:**  
   - Implements **Contrastive Decoding** and **Speculative Sampling** for enhanced output quality.  

---

## **Dataset List**  
DariushGPT is trained and evaluated using the following datasets:  

1. **OSCAR (Open Super-large Crawled ALMAnaCH coRpus)**  
   - **Link:** [HuggingFace Datasets - OSCAR](https://huggingface.co/datasets/oscar)  
   - **Usage:** Fundamental training for general Persian language understanding.  

2. **Persian Wikipedia Dump**  
   - **Link:** [Wikipedia Dumps](https://dumps.wikimedia.org/fawiki/)  
   - **Usage:** Enhancing the model’s knowledge in diverse topics.  

3. **Divan-e-Hafez (Hafez Poetry Collection)**  
   - **Link:** [GitHub - Persian Poetry Corpus](https://github.com/persiannlp/persian-poetry-corpus)  
   - **Usage:** Specialized training for Persian poetry generation.  

4. **SnappFood! Reviews**  
   - **Link:** [Kaggle Dataset](https://www.kaggle.com/datasets/snappfood/restaurant-comments)  
   - **Usage:** Improving sentiment analysis capabilities.  

5. **Digikala User Reviews**  
   - **Link:** [GitHub - Digikala Dataset](https://github.com/persiannlp/digikala-user-reviews)  
   - **Usage:** Training the model for sentiment analysis tasks.  

---

## **Related Projects**  
DariushGPT is inspired by and utilizes concepts from the following open-source projects:  

1. **ParsBERT**  
   - **Link:** [GitHub - ParsBERT](https://github.com/persiannlp/parsbert)  
   - **Usage:** Enhancing architecture and data preprocessing.  

2. **Persian NLP Toolkit**  
   - **Link:** [GitHub - Persian NLP](https://github.com/persiannlp/persian-nlp)  
   - **Usage:** Providing useful tools for Persian language processing.  

3. **Hafez-GPT**  
   - **Link:** [GitHub - Hafez-GPT](https://github.com/mehrdad-dev/Hafez-GPT)  
   - **Usage:** Inspiration for Persian poetry generation.  

4. **HuggingFace Transformers**  
   - **Link:** [GitHub - Transformers](https://github.com/huggingface/transformers)  
   - **Usage:** Implementing advanced Transformer-based architectures.  

5. **GPT-NeoX**  
   - **Link:** [GitHub - GPT-NeoX](https://github.com/EleutherAI/gpt-neox)  
   - **Usage:** Inspiration for scaling the model.  

---

## **Installation and Setup**  
To install and run DariushGPT, follow these steps:  

1. **Install dependencies:**  
   ```bash
   pip install torch transformers datasets faiss-cpu xformers deepspeed
   ```

2. **Download the model and datasets:**  
   ```bash
   git clone https://github.com/yourusername/DariushGPT.git
   cd DariushGPT
   ```

3. **Run the model:**  
   ```python
   from dariushgpt import DariushGPT, PersianTokenizer

   tokenizer = PersianTokenizer()
   model = DariushGPT(config)
   output = model.generate("سلام دنیا!")
   print(output)
   ```

---

## **Usage Examples**  

### **1. Text Generation:**  
```python
prompt = "In the name of God, the Most Gracious, the Most Merciful"
output = model.generate(prompt, max_length=50)
print(output)
```

### **2. Poetry Generation:**  
```python
poem = model.generate_poem(bahr="hazaj", rhyme="ar")
print(poem)
```

### **3. Sentiment Analysis:**  
```python
sentiment = model.analyze_sentiment("This movie was absolutely amazing!")
print(sentiment)  # Positive
```

### **4. Automatic Translation:**  
```python
translated = model.translate("سلام دنیا!")
print(translated)  # Hello World!
```

---

## **License**  
This project is released under the **MIT License**. For more details, see the [LICENSE](LICENSE) file.  

```markdown
MIT License

Copyright (c) 2025 hosein davod abadi farahani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
...
```

---

## **Contributing**  
We welcome contributions! To contribute, follow these steps:  

1. **Fork the repository.**  
2. **Create a new branch:**  
   ```bash
   git checkout -b feature/YourFeatureName
   ```  
3. **Commit your changes:**  
   ```bash
   git commit -m "Add YourFeatureName"
   ```  
4. **Push the changes:**  
   ```bash
   git push origin feature/YourFeatureName
   ```  
5. **Open a Pull Request.**  

---

## **Contact**  
For questions or suggestions, feel free to reach out via email:  
📧 **Email:** kinhofcod4242@gmail.com  

---

**Build the future of Persian language processing with DariushGPT!** 🚀
