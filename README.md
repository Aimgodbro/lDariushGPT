### **README.md**

---

# **DariushGPT**  
**یک مدل هوش مصنوعی چندمنظوره برای پردازش زبان فارسی**  
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)  
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)  
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)  

---

## **معرفی**  
**DariushGPT** یک مدل هوش مصنوعی پیشرفته برای پردازش زبان فارسی است که بر پایه معماری **Transformer** ساخته شده است. این مدل قابلیت‌های متنوعی از جمله تولید متن، تحلیل احساسات، تولید شعر، استدلال زنجیره‌ای (Chain-of-Thought) و ترجمه خودکار را ارائه می‌دهد. DariushGPT با بهره‌گیری از تکنیک‌های نوین مانند **Mixture of Experts (MoE)**، **Rotary Positional Embeddings (RoPE)** و **Retrieval-Augmented Generation (RAG)**، به یک ابزار قدرتمند برای کاربردهای مختلف تبدیل شده است.

---

## **ویژگی‌های کلیدی**  
✅ **چندمنظوره بودن:**  
   - تولید متن، شعر، تحلیل احساسات، ترجمه خودکار و استدلال زنجیره‌ای.  

✅ **معماری پیشرفته:**  
   - استفاده از **Transformer** با **FlashAttention** و **RoPE**.  
   - پیاده‌سازی **Mixture of Experts (MoE)** برای مدیریت تخصص‌های مختلف.  

✅ **یادگیری تقویتی:**  
   - پشتیبانی از **Reinforcement Learning from Human Feedback (RLHF)** و **Direct Preference Optimization (DPO)**.  

✅ **چندوجهیتی:**  
   - یکپارچه‌سازی با **CLIP** (پردازش تصویر) و **ASR** (پردازش صوت).  

✅ **سیستم RAG:**  
   - دسترسی به دانش خارجی با استفاده از **FAISS** برای بازیابی اطلاعات.  

✅ **تولید متن هوشمند:**  
   - استفاده از **Contrastive Decoding** و **Speculative Sampling** برای بهبود کیفیت خروجی.  

---

## **لیست دیتاست‌ها**  
مدل DariushGPT از دیتاست‌های زیر برای آموزش و ارزیابی استفاده می‌کند:  

1. **OSCAR (Open Super-large Crawled ALMAnaCH coRpus)**  
   - **لینک:** [HuggingFace Datasets - OSCAR](https://huggingface.co/datasets/oscar)  
   - **کاربرد:** آموزش پایه برای درک عمومی زبان فارسی.  

2. **Persian Wikipedia Dump**  
   - **لینک:** [Wikipedia Dumps](https://dumps.wikimedia.org/fawiki/)  
   - **کاربرد:** بهبود دانش عمومی مدل در موضوعات متنوع.  

3. **Divan-e-Hafez (دیوان حافظ)**  
   - **لینک:** [GitHub - Persian Poetry Corpus](https://github.com/persiannlp/persian-poetry-corpus)  
   - **کاربرد:** آموزش تخصصی برای تولید شعر فارسی.  

4. **SnappFood! Reviews**  
   - **لینک:** [Kaggle Dataset](https://www.kaggle.com/datasets/snappfood/restaurant-comments)  
   - **کاربرد:** بهبود تحلیل احساسات.  

5. **Digikala User Reviews**  
   - **لینک:** [GitHub - Digikala Dataset](https://github.com/persiannlp/digikala-user-reviews)  
   - **کاربرد:** آموزش مدل برای تحلیل احساسات.  

---

## **پروژه‌های مرتبط**  
DariushGPT از پروژه‌های متن‌باز زیر الهام گرفته و استفاده می‌کند:  

1. **ParsBERT**  
   - **لینک:** [GitHub - ParsBERT](https://github.com/persiannlp/parsbert)  
   - **کاربرد:** بهبود معماری و پیش‌پردازش داده‌ها.  

2. **Persian NLP Toolkit**  
   - **لینک:** [GitHub - Persian NLP](https://github.com/persiannlp/persian-nlp)  
   - **کاربرد:** ابزارهای کمکی برای پردازش زبان فارسی.  

3. **Hafez-GPT**  
   - **لینک:** [GitHub - Hafez-GPT](https://github.com/mehrdad-dev/Hafez-GPT)  
   - **کاربرد:** الهام‌گیری برای بخش شعر مدل.  

4. **HuggingFace Transformers**  
   - **لینک:** [GitHub - Transformers](https://github.com/huggingface/transformers)  
   - **کاربرد:** پیاده‌سازی معماری‌های پیشرفته.  

5. **GPT-NeoX**  
   - **لینک:** [GitHub - GPT-NeoX](https://github.com/EleutherAI/gpt-neox)  
   - **کاربرد:** الهام‌گیری برای مقیاس‌پذیری.  

---

## **نصب و راه‌اندازی**  
برای نصب و اجرای DariushGPT، مراحل زیر را دنبال کنید:  

1. **نصب وابستگی‌ها:**  
   ```bash
   pip install torch transformers datasets faiss-cpu xformers deepspeed
   ```

2. **دانلود مدل و دیتاست‌ها:**  
   ```bash
   git clone https://github.com/yourusername/DariushGPT.git
   cd DariushGPT
   ```

3. **اجرای مدل:**  
   ```python
   from dariushgpt import DariushGPT, PersianTokenizer

   tokenizer = PersianTokenizer()
   model = DariushGPT(config)
   output = model.generate("سلام دنیا!")
   print(output)
   ```

---

## **مثال‌های کاربردی**  

### **1. تولید متن:**  
```python
prompt = "به نام خداوند جان و خرد"
output = model.generate(prompt, max_length=50)
print(output)
```

### **2. تولید شعر:**  
```python
poem = model.generate_poem(bahr="hazaj", rhyme="ar")
print(poem)
```

### **3. تحلیل احساسات:**  
```python
sentiment = model.analyze_sentiment("این فیلم واقعا عالی بود!")
print(sentiment)  # مثبت
```

### **4. ترجمه خودکار:**  
```python
translated = model.translate("سلام دنیا!")
print(translated)  # Hello World!
```

---

## **مجوز (License)**  
این پروژه تحت مجوز **MIT** منتشر شده است. برای اطلاعات بیشتر به فایل [LICENSE](LICENSE) مراجعه کنید.  

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

## **همکاری و مشارکت**  
ما از مشارکت‌های شما استقبال می‌کنیم! برای همکاری، مراحل زیر را دنبال کنید:  

1. ریپازیتوری را **Fork** کنید.  
2. یک **Branch** جدید ایجاد کنید:  
   ```bash
   git checkout -b feature/YourFeatureName
   ```  
3. تغییرات خود را **Commit** کنید:  
   ```bash
   git commit -m "Add YourFeatureName"
   ```  
4. تغییرات را **Push** کنید:  
   ```bash
   git push origin feature/YourFeatureName
   ```  
5. یک **Pull Request** باز کنید.  

---

## **تماس با ما**  
برای هرگونه سوال یا پیشنهاد، می‌توانید با ما از طریق ایمیل زیر در تماس باشید:  
📧 **Email:** kinhofcod4242@gmail.com 

---

**با DariushGPT، آینده‌ی پردازش زبان فارسی را بسازید!** 🚀
