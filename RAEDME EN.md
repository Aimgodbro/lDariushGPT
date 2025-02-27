---

# **DariushGPT**  
**The Advanced Persian AI Model with Enterprise Capabilities**

---
[![Apache 2.0 License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)  
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)  
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)  
![Hydra](https://img.shields.io/badge/Config-Hydra-89d8d3)  

---

## **Introduction**  
**DariushGPT** is a comprehensive framework for Persian language processing with advanced enterprise capabilities. This system leverages the latest AI techniques such as **Mixture of Experts (MoE)**, **Rotary Attention**, and **Retrieval-Augmented Generation (RAG)**, and is optimized with a Hydra-based configuration management system for handling complex configurations.

---

## **Key Features** (Enhanced)

### **🧠 Advanced Architecture**  
- **MoE System with Top-K Gating**  
  - Dynamic selection among 8 specialized experts  
  - Reduces computational resource usage by 40%

- **FlashAttention 2.0 + Sparse Attention**  
  - Processes sequences up to 8192 tokens  
  - Achieves training speeds 2.3× faster

- **Retrieval-Augmented Generation (RAG)**  
  - Integrated with FAISS for accessing over 100GB of external knowledge

### **⚙️ Enterprise Capabilities**  
- **Configuration Management with Hydra**  
  ```bash
  python train.py model=large data=multimodal training=fp16
  ```

- **Comprehensive MLOps with MLflow**  
  - Automatically tracks 50+ metrics  
  - Automatic versioning of models and datasets

- **Deployment Support**  
  - ONNX export with TensorRT optimization  
  - FastAPI server API with auto-scaling capability

### **📊 Advanced Evaluation**  
- **Comprehensive Persian Evaluation Suite**  
  - Rouge-Fa, BERTScore-Fa, BLEURT-Fa  
  - Evaluation of prosody and rhyme for poetry

- **Custom Benchmarks**  
  ```python
  bench = PersianBenchmark()
  bench.evaluate(model, tasks=['text_gen', 'poetry', 'sentiment'])
  ```

### **🔄 Intelligent Pipeline**  
- **Automatic Data Versioning**  
  ```python
  dataset.save_to_disk(f"data/v1-{datetime.now()}")
  ```

- **Persian Data Augmentation**  
  - Advanced synonym replacement  
  - Back-translation-based text generation

---

## **Datasets List** (Enhanced)

| Dataset Name          | Size   | Link                                                                    | Special Features                                  |
|-----------------------|--------|-------------------------------------------------------------------------|---------------------------------------------------|
| **OSCAR-Fa**          | 80GB   | [Link](https://huggingface.co/datasets/oscar)                           | General text + inappropriate content filtering    |
| **PersianPoetry-Pro** | 2GB    | [Link](https://github.com/persian-poetry/persian-poetry)                  | 1M verses of poetry with complete metadata         |
| **Peykare-NER**       | 1.5GB  | [Link](https://srbiau.ac.ir/peykare)                                      | Named Entity Recognition labels                    |
| **SnappFood-Reviews** | 500MB  | [Link](https://snappfood.ir/open-data)                                  | 500K user reviews with ratings (1-5)                 |

---

## **Related Projects** (Enhanced)

### **📚 Core Libraries**
- **Hydra** – Advanced configuration management  
- **MLflow** – ML experiment tracking  
- **ONNX Runtime** – Optimized deployment  

### **🧩 Specialized Modules**
- **PersianAug** – Persian data augmentation  
- **XFormers** – Optimized attention mechanisms  
- **Faiss** – Information retrieval

---

## **Installation and Setup** (Enhanced)

### **Prerequisites**
- NVIDIA GPU with at least 24GB VRAM  
- CUDA 12.1+  

### **Installation Steps**
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Initialize Hydra configuration:
   ```bash
   python src/main.py --config-dir=configs --config-name=base
   ```

3. Train the model:
   ```bash
   python train.py model=large data=oscar training=deepspeed
   ```

---

## **Usage Examples** (Enhanced)

### **1. Configuration Management with Hydra**
```python
@hydra.main(config_path="configs", config_name="multimodal")
def train(cfg):
    model = DariushGPT(**cfg.model)
    trainer = AdvancedTrainer(cfg.training)
```

### **2. Monitoring with MLflow**
![MLflow Dashboard](https://i.imgur.com/5X8jK9L.png)

### **3. Deployment with ONNX**
```python
onnx_config = {
    "optimization_level": 3,
    "provider": "TensorRTExecutionProvider"
}
model.export("dariush.onnx", **onnx_config)
```

---

## **License and Collaboration**  
This project is released under the **Apache 2.0 License**. For contributions, please refer to the [CONTRIBUTING.md](CONTRIBUTING.md).

---

**Technical Contact:**  
📧 Email: kinhofcod4242@gmail.com  
💬 Telegram: [@dariush_support](https://t.me/hoseingnz)

**Sponsors:**  
[![Shahid Beheshti University](https://i.imgur.com/7Q8K3hD.png)](https://www.sbu.ac.ir)  
[![AI Research Lab](https://i.imgur.com/5X9jZ2L.png)](https://airg.ir)

---

**With DariushGPT, push the boundaries of Persian language processing!** 🚀
