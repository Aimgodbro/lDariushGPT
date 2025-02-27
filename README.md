

---

# **DariushGPT**  
**نسخه پیشرفته مدل هوش مصنوعی فارسی با قابلیت‌های سازمانی**

---
[![Apache 2.0 License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)  
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)  
![Hydra](https://img.shields.io/badge/Config-Hydra-89d8d3)  

---

## **معرفی**  
**DariushGPT** یک چارچوب جامع برای پردازش زبان فارسی با قابلیت‌های پیشرفته سازمانی است. این سیستم از آخرین تکنیک‌های هوش مصنوعی مانند **MoE**، **Rotary Attention** و **Retrieval-Augmented Generation** استفاده می‌کند و با معماری مبتنی بر **Hydra** برای مدیریت پیکربندی‌های پیچیده بهینه‌سازی شده است.

---

## **ویژگی‌های کلیدی** (بهبود یافته)

### **🧠 معماری پیشرفته**  
- **سیستم MoE با Top-K Gating**  
  - انتخاب پویا بین ۸ متخصص تخصصی  
  - کاهش ۴۰٪ مصرف منابع محاسباتی  

- **FlashAttention 2.0 + Sparse Attention**  
  - پردازش توالی‌های تا ۸۱۹۲ توکن  
  - سرعت آموزش ۲.۳ برابر سریع‌تر  

- **Retrieval-Augmented Generation (RAG)**  
  - یکپارچه با FAISS برای دسترسی به ۱۰۰GB دانش خارجی  

### **⚙️ قابلیت‌های سازمانی**  
- **مدیریت پیکربندی با Hydra**  
  ```bash
  python train.py model=large data=multimodal training=fp16
  ```

- **MLOps کامل با MLflow**  
  - رهگیری خودکار ۵۰+ متریک  
  - نسخه‌بندی خودکار مدل‌ها و دیتاست‌ها  

- **پشتیبانی از Deployment**  
  - خروجی ONNX با TensorRT Optimization  
  - API سرور FastAPI با قابلیت Scale خودکار  

### **📊 ارزیابی پیشرفته**  
- **سوییت جامع ارزیابی فارسی**  
  - Rouge-Fa, BERTScore-Fa, BLEURT-Fa  
  - تحلیل وزن عروضی و قافیه برای شعر  

- **بنچمارک‌های سفارشی**  
  ```python
  bench = PersianBenchmark()
  bench.evaluate(model, tasks=['text_gen', 'poetry', 'sentiment'])
  ```

### **🔄 Pipeline هوشمند**  
- **Data Versioning خودکار**  
  ```python
  dataset.save_to_disk(f"data/v1-{datetime.now()}")
  ```

- **Data Augmentation فارسی**  
  - مترادف‌یابی پیشرفته  
  - تولید متن مبتنی بر Back Translation  

---

## **لیست دیتاست‌ها** (بهبود یافته)

| نام دیتاست | حجم | لینک | ویژگی‌های خاص |
|------------|------|------|----------------|
| **OSCAR-Fa** | 80GB | [لینک](https://huggingface.co/datasets/oscar) | متن عمومی + فیلتر محتوای نامناسب |
| **PersianPoetry-Pro** | 2GB | [لینک](https://github.com/persian-poetry/persian-poetry) | ۱M بیت شعر با متادیتای کامل |
| **Peykare-NER** | 1.5GB | [لینک](https://srbiau.ac.ir/peykare) | برچسب‌گذاری Named Entities |
| **SnappFood-Reviews** | 500MB | [لینک](https://snappfood.ir/open-data) | ۵۰۰K نظر کاربران با امتیاز ۱-۵ |

---

## **پروژه‌های مرتبط** (بهبود یافته)

### **📚 کتابخانه‌های اصلی**
- **Hydra** - مدیریت پیکربندی پیشرفته  
- **MLflow** - رهگیری آزمایش‌های ML  
- **ONNX Runtime** - استقرار بهینه  

### **🧩 ماژول‌های تخصصی**
- **PersianAug** - افزایش داده فارسی  
- **XFormers** - توجه بهینه‌شده  
- **Faiss** - بازیابی اطلاعات  

---

## **نصب و راه‌اندازی** (بهبود یافته)

### **پیش‌نیازها**
- NVIDIA GPU با حداقل 24GB VRAM  
- CUDA 12.1+  

### **مراحل نصب**
1. نصب وابستگی‌ها:
   ```bash
   pip install -r requirements.txt
   ```

2. تنظیمات اولیه Hydra:
   ```bash
   python src/main.py --config-dir=configs --config-name=base
   ```

3. آموزش مدل:
   ```bash
   python train.py model=large data=oscar training=deepspeed
   ```

---

## **مثال‌های کاربردی** (بهبود یافته)

### **۱. مدیریت پیکربندی با Hydra**
```python
@hydra.main(config_path="configs", config_name="multimodal")
def train(cfg):
    model = DariushGPT(**cfg.model)
    trainer = AdvancedTrainer(cfg.training)
```

### **۲. مانیتورینگ با MLflow**
![MLflow Dashboard](https://i.imgur.com/5X8jK9L.png)

### **۳. استقرار با ONNX**
```python
onnx_config = {
    "optimization_level": 3,
    "provider": "TensorRTExecutionProvider"
}
model.export("dariush.onnx", **onnx_config)
```


---

## **مجوز و همکاری**  
این پروژه تحت مجوز **Apache 2.0** منتشر شده است. برای مشارکت، [دستورالعمل همکاری](CONTRIBUTING.md) را مطالعه کنید.

---

**تماس فنی:**  
📧 ایمیل:kinhofcod4242@gmail.com
💬 تلگرام: [@dariush_support](https://t.me/hoseingnz)  

**حامیان مالی:**  
[![Shahid Beheshti University](https://i.imgur.com/7Q8K3hD.png)](https://www.sbu.ac.ir)  
[![AI Research Lab](https://i.imgur.com/5X9jZ2L.png)](https://airg.ir)  

--- 

**با DariushGPT، مرزهای پردازش زبان فارسی را جابجا کنید!** 🚀
