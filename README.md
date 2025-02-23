

```markdown
# DariushGPT: مدل زبانی پیشرفته برای زبان فارسی

DariushGPT یک مدل زبانی مبتنی بر ترانسفورمر است که برای زبان فارسی طراحی شده است. این مدل قابلیت‌های متنوعی مانند تولید متن، تولید شعر و تحلیل احساسات را ارائه می‌دهد و برای استفاده در تحقیقات و کاربردهای عملی بهینه‌سازی شده است.

---

## ویژگی‌های کلیدی

- **تولید متن:** توانایی تولید متن‌های روان و پیوسته به زبان فارسی.
- **تولید شعر:** قابلیت تولید شعر با رعایت وزن عروضی و قافیه.
- **تحلیل احساسات:** تشخیص احساسات متن (مثبت، منفی، خنثی).
- **پشتیبانی از Hugging Face Datasets:** امکان استفاده از دیتاست‌های بزرگ فارسی موجود در Hugging Face.
- **بهینه‌سازی برای سخت‌افزارهای مختلف:** پشتیبانی از CUDA، MPS (برای Apple Silicon) و CPU.
- **آموزش ترکیبی دقت-مختلط (AMP):** افزایش سرعت آموزش با حفظ دقت.
- **ذخیره خودکار بهترین مدل:** ذخیره مدل با کمترین Loss در طول آموزش.

---

## معماری مدل

DariushGPT از یک معماری **ترانسفورمر** با ویژگی‌های زیر استفاده می‌کند:

- **لایه‌های ترانسفورمر:** ۶ لایه با ۸ هد توجه.
- **اندازه امبدینگ:** ۵۱۲ بعدی.
- **اندازه واژگان:** ۳۰٬۰۰۰ توکن.
- **طول دنباله:** حداکثر ۵۱۲ توکن.
- **هدهای تخصصی:** هدهای جداگانه برای تولید متن، شعر و تحلیل احساسات.

---

## نحوه استفاده

### ۱. نصب پیش‌نیازها

برای اجرای کد، ابتدا کتابخانه‌های مورد نیاز را نصب کنید:

```bash
pip install torch tokenizers datasets tqdm
```

### ۲. اجرای آموزش مدل

برای آموزش مدل، کافی است فایل اصلی را اجرا کنید:

```bash
python dariush.py
```

### ۳. استفاده از مدل آموزش‌دیده

پس از آموزش، مدل در فایل `best_dariush_model.pt` ذخیره می‌شود. می‌توانید از این مدل برای تولید متن، شعر یا تحلیل احساسات استفاده کنید.

```python
import torch

# بارگیری مدل
checkpoint = torch.load("best_dariush_model.pt", map_location="cpu")
model = DariushGPT(checkpoint["config"], checkpoint["tokenizer"])
model.load_state_dict(checkpoint["model_state"])

# تولید متن
text = model.generate([model.tokenizer.special_tokens["[BOS]"]])
print(model.tokenizer.decode(text))

# تولید شعر
poem = model.generate_poem(bahr="hazaj", rhyme="ar")
print(model.tokenizer.decode(poem))

# تحلیل احساسات
sentiment = model.analyze_sentiment("این یک شاهکار است!")
print(sentiment)  # مثبت
```

---

## دیتاست‌ها

این پروژه از دیتاست‌های زیر پشتیبانی می‌کند:

- **[OSCAR](https://huggingface.co/datasets/oscar):** دیتاست عمومی فارسی از Hugging Face.
- **[GANJUR](https://github.com/persiannlp/ganjoor):** دیتاست شعر کلاسیک فارسی.
- **[Sentiment Analysis](https://github.com/phosseini/sentiment-persian):** دیتاست تحلیل احساسات فارسی.

---

## بهینه‌سازی‌ها

- **پشتیبانی از MPS:** بهینه‌سازی برای کارت‌های گرافیک Apple Silicon.
- **Gradient Checkpointing:** کاهش مصرف حافظه در طول آموزش.
- **AMP:** آموزش ترکیبی دقت-مختلط برای افزایش سرعت.

---

## نمونه‌های خروجی

### تولید متن
```
زندگی زیباست چون هر لحظه آن پر از شگفتی‌های ناشناخته است...
```

### تولید شعر
```
به نام خداوند جان و خرد
کزین برتر اندیشه برنگذرد
به پیشگاه او سپر اندازم
که از الطافش برخوردارم
```

### تحلیل احساسات
```
متن: "این فیلم واقعا عالی بود!"
احساسات: مثبت
```

---

## پروژه‌های مرتبط

این پروژه از کتابخانه‌ها و دیتاست‌های زیر استفاده کرده است:

- **[PyTorch](https://github.com/pytorch/pytorch):** چارچوب یادگیری عمیق.
- **[Hugging Face Datasets](https://github.com/huggingface/datasets):** کتابخانه برای بارگیری و مدیریت دیتاست‌ها.
- **[Tokenizers](https://github.com/huggingface/tokenizers):** کتابخانه برای توکنایز کردن متن.
- **[GANJUR](https://github.com/persiannlp/ganjoor):** دیتاست شعر فارسی.
- **[Persian Sentiment Analysis](https://github.com/phosseini/sentiment-persian):** دیتاست تحلیل احساسات فارسی.

---

## مشارکت

اگر می‌خواهید در توسعه این پروژه مشارکت کنید، لطفاً مراحل زیر را دنبال کنید:

1. پروژه را Fork کنید.
2. یک Branch جدید ایجاد کنید (`git checkout -b feature/YourFeatureName`).
3. تغییرات خود را Commit کنید (`git commit -m 'Add some feature'`).
4. تغییرات را Push کنید (`git push origin feature/YourFeatureName`).
5. یک Pull Request باز کنید.

---

## لایسنس

این پروژه تحت لایسنس [MIT](LICENSE)

---

## تماس با ما

اگر سوالی دارید یا به کمک نیاز دارید، می‌توانید از طریق ایمیل با ما تماس بگیرید:  
📧 kinhofcod4242@gmail.com

---

**با DariushGPT، قدرت زبان فارسی را کشف کنید!** 🚀
```



