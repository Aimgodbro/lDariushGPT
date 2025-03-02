# GodModeDariushCosmic: The most advanced transformer model ever conceived
# Copyright (c) 2025 hosein davod abadi farahani & cosmic enhancements

import jax
import jax.numpy as jnp
import haiku as hk
from jax import config as jax_config
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax.lax import with_sharding_constraint as pjit_sharding_constraint
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import load_dataset
from typing import Optional, List, Dict, Any, Tuple
from tqdm import tqdm
import functools
import logging
import optax
import numpy as np
from dataclasses import dataclass
import jax.tree_util as tree_util
import threading
import queue
import os
import time
import json
from concurrent.futures import ThreadPoolExecutor

jax_config.update("jax_spmd_mode", "allow_all")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. تنظیمات کیهانی
@dataclass
class CosmicConfig:
    vocab_size: int = 131072  # برای پوشش زبانی وسیع
    emb_size: int = 8192      # تعبیه عظیم
    num_q_heads: int = 128    # GQA پیشرفته
    num_kv_heads: int = 16    # بهینه‌سازی حافظه
    key_size: int = 128
    num_layers: int = 64      # عمیق‌ترین مدل
    num_experts: int = 64     # MoE کیهانی
    num_selected_experts: int = 8
    widening_factor: float = 4.5
    max_seq_len: int = 16384  # برای متن‌های عظیم
    init_scale: float = 0.01
    dropout_rate: float = 0.1
    sparse_factor: int = 4    # برای توجه پراکنده
    data_axis: str = "data"
    model_axis: str = "model"
    expert_axis: str = "expert"
    shard_activations: bool = True
    use_swiglu: bool = True
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    batch_size: int = 16
    num_micro_batches: int = 4
    learning_rate: float = 6e-5
    warmup_steps: int = 2000
    total_steps: int = 50000
    special_tokens: Dict[str, int] = field(default_factory=lambda: {
        "[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3, "[CLS]": 4, "[SEP]": 5
    })

    def partition_rules(self):
        return [
            # تعبیه‌ها
            (("embedding", "w"), P(None, "data", "model")),
            # توجه چندسر
            (("multi_head_attention", "(query|key|value)", "w"), P("data", "model")),
            (("multi_head_attention", "linear", "w"), P("model", "data")),
            # MoE
            (("moe", "router", "w"), P("data", "expert")),
            (("moe", "expert", "w"), P("expert", "data", "model")),
            (("moe", "expert_out", "w"), P("expert", "model", "data")),
            # نرمال‌سازی
            (("rms_norm", "scale"), P(None)),
            # خروجی
            (("output", "w"), P("model", "data")),
            # حافظه KV
            (("kv_cache", "k"), P("data", "model")),
            (("kv_cache", "v"), P("data", "model")),
        ]

config = CosmicConfig()

# 2. توکنایزر کیهانی
class CosmicTokenizer:
    def __init__(self):
        self.tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        self.special_tokens = config.special_tokens
        self.cache = {}

    def train(self, data_path="oscar_fa"):
        logger.info("Training cosmic tokenizer...")
        dataset = load_dataset("oscar", "unshuffled_deduplicated_fa", split="train[:20%]")
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=config.vocab_size,
            special_tokens=list(self.special_tokens.keys()),
            min_frequency=2,
            show_progress=True
        )
        self.tokenizer.train_from_iterator(dataset["text"], trainer=trainer)
        self.tokenizer.save("cosmic_dariush_tokenizer.json")
        logger.info("Tokenizer trained and saved.")

    def encode(self, text):
        if text in self.cache:
            return self.cache[text]
        tokens = jnp.array(self.tokenizer.encode(text).ids)
        self.cache[text] = tokens
        return tokens

    def decode(self, tokens):
        return self.tokenizer.decode(tokens.tolist())

    def pad(self, sequences, max_len):
        return jnp.array([seq[:max_len] + [self.special_tokens["[PAD]"]] * (max_len - len(seq)) 
                         for seq in sequences])

    def batch_encode(self, texts):
        return self.pad([self.encode(text) for text in texts], config.max_seq_len)

# 3. دیتالودر توزیع‌شده
class CosmicDataLoader:
    def __init__(self, tokenizer, batch_size, num_workers=4):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.queue = queue.Queue(maxsize=100)
        self.dataset = load_dataset("oscar", "unshuffled_deduplicated_fa", split="train[:20%]")["text"]
        self.running = False

    def start(self):
        self.running = True
        self.workers = []
        for _ in range(self.num_workers):
            worker = threading.Thread(target=self._worker_fn)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def stop(self):
        self.running = False
        for worker in self.workers:
            worker.join()

    def _worker_fn(self):
        while self.running:
            batch_texts = np.random.choice(self.dataset, self.batch_size, replace=False)
            batch = {
                "input_ids": self.tokenizer.batch_encode(batch_texts),
                "labels": self.tokenizer.batch_encode(batch_texts),
                "mask": (self.tokenizer.batch_encode(batch_texts) != self.tokenizer.special_tokens["[PAD]"])[:, None, None, :]
            }
            self.queue.put(batch)

    def __iter__(self):
        return self

    def __next__(self):
        if not self.running:
            raise StopIteration
        return self.queue.get()

# 4. RMSNorm کیهانی
class CosmicRMSNorm(hk.Module):
    def __init__(self, emb_size, eps=1e-6, name="rms_norm"):
        super().__init__(name=name)
        self.emb_size = emb_size
        self.eps = eps

    def __call__(self, x):
        scale = hk.get_parameter("scale", [self.emb_size], init=jnp.ones)
        scale = pjit_sharding_constraint(scale, P(None))
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(variance + self.eps) * scale
        return normed.astype(jnp.bfloat16)

# 5. Rotary Embedding کیهانی
class CosmicRotaryEmbedding(hk.Module):
    def __init__(self, dim, base=10000, max_seq_len=16384, name="rotary_emb"):
        super().__init__(name=name)
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def __call__(self, x, offset=0):
        seq_len = x.shape[1]
        pos = jnp.arange(seq_len, dtype=jnp.float32) + offset
        angles = pos[:, None] * self.inv_freq[None, :]
        sin_val = jnp.sin(angles)
        cos_val = jnp.cos(angles)
        x1, x2 = x[..., :self.dim//2], x[..., self.dim//2:]
        x_rot = jnp.concatenate([-x2, x1], axis=-1)
        return x * cos_val + x_rot * sin_val

# 6. SwiGLU کیهانی
class CosmicSwiGLU(hk.Module):
    def __init__(self, hidden_size, name="swiglu"):
        super().__init__(name=name)
        self.hidden_size = hidden_size

    def __call__(self, x):
        w1 = hk.Linear(self.hidden_size, name="w1")
        w2 = hk.Linear(self.hidden_size, name="w2")
        return jax.nn.silu(w1(x)) * w2(x)

# 7. Flash Attention (بهینه‌سازی توجه)
class CosmicFlashAttention(hk.Module):
    def __init__(self, num_heads, key_size, block_size=128, name="flash_attention"):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.block_size = block_size

    def __call__(self, q, k, v, mask=None):
        batch, seq_len, _ = q.shape
        q = q.reshape(batch, seq_len, self.num_heads, self.key_size)
        k = k.reshape(batch, seq_len, self.num_heads, self.key_size)
        v = v.reshape(batch, seq_len, self.num_heads, self.key_size)

        def block_attention(q_block, k_block, v_block, mask_block):
            attn_logits = jnp.einsum("...hd,...kd->...hk", q_block, k_block) / jnp.sqrt(self.key_size)
            if mask_block is not None:
                attn_logits = jnp.where(mask_block, attn_logits, -1e30)
            attn_weights = jax.nn.softmax(attn_logits)
            return jnp.einsum("...hk,...kd->...hd", attn_weights, v_block)

        num_blocks = seq_len // self.block_size
        q_blocks = q.reshape(batch, num_blocks, self.block_size, self.num_heads, self.key_size)
        k_blocks = k.reshape(batch, num_blocks, self.block_size, self.num_heads, self.key_size)
        v_blocks = v.reshape(batch, num_blocks, self.block_size, self.num_heads, self.key_size)
        mask_blocks = mask.reshape(batch, 1, num_blocks, self.block_size) if mask is not None else None

        outputs = jax.vmap(block_attention)(q_blocks, k_blocks, v_blocks, mask_blocks)
        return outputs.reshape(batch, seq_len, -1)

# 8. MoE کیهانی
class CosmicRouter(hk.Module):
    def __init__(self, num_experts, num_selected_experts, name="router"):
        super().__init__(name=name)
        self.num_experts = num_experts
        self.num_selected_experts = num_selected_experts

    def __call__(self, inputs):
        w = hk.get_parameter("w", [inputs.shape[-1], self.num_experts], init=hk.initializers.TruncatedNormal(stddev=0.02))
        w = pjit_sharding_constraint(w, P("data", "expert"))
        logits = jnp.dot(inputs.astype(jnp.float32), w)
        noise = jax.random.gumbel(jax.random.PRNGKey(0), logits.shape) * 0.05
        probs = jax.nn.softmax(logits + noise)
        gates, indices = jax.lax.top_k(probs, self.num_selected_experts)
        return gates, indices

class CosmicMoELayer(hk.Module):
    def __init__(self, config, mesh, name="moe"):
        super().__init__(name=name)
        self.config = config
        self.mesh = mesh
        self.router = CosmicRouter(config.num_experts, config.num_selected_experts)

    def __call__(self, inputs):
        gates, indices = self.router(inputs)
        expert_outputs = []

        def expert_fn(x):
            w = hk.Linear(int(self.config.widening_factor * self.config.emb_size), name="expert")
            w_out = hk.Linear(self.config.emb_size, name="expert_out")
            if self.config.use_swiglu:
                return w_out(CosmicSwiGLU(self.config.emb_size)(x))
            return w_out(jax.nn.gelu(w(x)))

        for _ in range(self.config.num_experts):
            expert_outputs.append(expert_fn(inputs))

        expert_outputs = jnp.stack(expert_outputs, axis=1)  # [batch, experts, seq, emb]

        @functools.partial(shard_map, mesh=self.mesh, in_specs=(P("data", None, "expert"), P("expert", "data", "model")),
                           out_specs=P("data", "model"), check_rep=False)
        def compute_expert_output(inputs, expert_outs):
            return jax.vmap(lambda x, idx: x[idx])(inputs, indices)

        selected_outputs = compute_expert_output(inputs, expert_outputs)
        return (selected_outputs * gates[..., None]).sum(axis=1)

# 9. توجه چندسر کیهانی
class CosmicMultiHeadAttention(hk.Module):
    def __init__(self, config, name="multi_head_attention"):
        super().__init__(name=name)
        self.config = config
        self.rotary = CosmicRotaryEmbedding(config.key_size)

    def __call__(self, x, mask=None, kv_cache=None):
        q_w = hk.Linear(self.config.num_q_heads * self.config.key_size, name="query")
        k_w = hk.Linear(self.config.num_kv_heads * self.config.key_size, name="key")
        v_w = hk.Linear(self.config.num_kv_heads * self.config.key_size, name="value")
        out_w = hk.Linear(self.config.emb_size, name="linear")

        q = q_w(x).reshape(*x.shape[:-1], self.config.num_q_heads, self.config.key_size)
        k = k_w(x).reshape(*x.shape[:-1], self.config.num_kv_heads, self.config.key_size)
        v = v_w(x).reshape(*x.shape[:-1], self.config.num_kv_heads, self.config.key_size)

        q = self.rotary(q)
        k = self.rotary(k)

        if kv_cache is not None:
            k = kv_cache["k"]
            v = kv_cache["v"]

        if self.config.use_flash_attention:
            flash_attn = CosmicFlashAttention(self.config.num_q_heads, self.config.key_size)
            attn_output = flash_attn(q, k, v, mask)
        else:
            attn_logits = jnp.einsum("...qhd,...khd->...hqk", q, k) / jnp.sqrt(self.config.key_size)
            if mask is not None:
                attn_logits = jnp.where(mask, attn_logits, -1e30)
            attn_weights = jax.nn.softmax(attn_logits)
            attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, v).reshape(*x.shape[:-1], -1)

        return out_w(attn_output), {"k": k, "v": v}

# 10. لایه کیهانی
class CosmicDariushLayer(hk.Module):
    def __init__(self, config, mesh, layer_idx, name="cosmic_layer"):
        super().__init__(name=f"{name}_{layer_idx}")
        self.config = config
        self.mesh = mesh
        self.attn = CosmicMultiHeadAttention(config)
        self.moe = CosmicMoELayer(config, mesh)
        self.norm1 = CosmicRMSNorm(config.emb_size)
        self.norm2 = CosmicRMSNorm(config.emb_size)
        self.dropout = hk.dropout if config.dropout_rate > 0 else lambda x: x

    def __call__(self, x, mask=None, kv_cache=None):
        if self.config.gradient_checkpointing:
            attn_out, new_cache = hk.checkpoint(lambda x: self.attn(self.norm1(x), mask, kv_cache))(x)
        else:
            attn_out, new_cache = self.attn(self.norm1(x), mask, kv_cache)
        x = x + self.dropout(attn_out, rate=self.config.dropout_rate, salt=jax.random.PRNGKey(layer_idx))
        moe_out = self.moe(self.norm2(x))
        x = x + self.dropout(moe_out, rate=self.config.dropout_rate, salt=jax.random.PRNGKey(layer_idx + 1))
        return x, new_cache

# 11. مدل اصلی کیهانی
class GodModeDariushCosmic(hk.Module):
    def __init__(self, config, mesh, name="godmode_dariush_cosmic"):
        super().__init__(name=name)
        self.config = config
        self.mesh = mesh
        self.embedding = hk.Embed(config.vocab_size, config.emb_size, name="embedding")
        self.layers = [CosmicDariushLayer(config, mesh, i) for i in range(config.num_layers)]
        self.norm = CosmicRMSNorm(config.emb_size)
        self.output = hk.Linear(config.vocab_size, name="output")

    def __call__(self, input_ids, mask=None, kv_cache=None):
        x = self.embedding(input_ids)
        x = pjit_sharding_constraint(x, P(self.config.data_axis, None, self.config.model_axis))
        new_kv_cache = [] if kv_cache is None else kv_cache

        for i, layer in enumerate(self.layers):
            x, layer_cache = layer(x, mask, new_kv_cache[i] if kv_cache else None)
            new_kv_cache.append(layer_cache)

        x = self.norm(x)
        logits = self.output(x)
        return logits, new_kv_cache

    def init_memory(self, batch_size, seq_len):
        return [{"k": jnp.zeros((batch_size, seq_len, self.config.num_kv_heads, self.config.key_size), dtype=jnp.bfloat16),
                 "v": jnp.zeros((batch_size, seq_len, self.config.num_kv_heads, self.config.key_size), dtype=jnp.bfloat16)}
                for _ in range(self.config.num_layers)]

    def generate(self, input_ids, max_len=200, temperature=0.7, top_k=40, beam_width=5):
        kv_cache = self.init_memory(input_ids.shape[0], input_ids.shape[1])
        beams = [(input_ids, 0.0, kv_cache)]  # (sequence, score, cache)

        for _ in range(max_len):
            new_beams = []
            for seq, score, cache in beams:
                logits, new_cache = self(seq, kv_cache=cache)
                next_logits = logits[:, -1, :] / temperature
                top_k_logits, top_k_tokens = jax.lax.top_k(next_logits, top_k)
                probs = jax.nn.softmax(top_k_logits)
                
                for i in range(top_k):
                    new_seq = jnp.concatenate([seq, top_k_tokens[:, i:i+1]], axis=1)
                    new_score = score + jnp.log(probs[:, i])
                    new_beams.append((new_seq, new_score, new_cache))

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            if jnp.all(beams[0][0][:, -1] == self.config.special_tokens["[EOS]"]):
                break

        return beams[0][0]

# 12. مدیریت چک‌پوینت
class CosmicCheckpointManager:
    def __init__(self, save_dir="checkpoints"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save(self, params, step):
        path = os.path.join(self.save_dir, f"step_{step}.jax")
        with open(path, "wb") as f:
            f.write(jax.tree_util.tree_flatten(params)[0])
        logger.info(f"Saved checkpoint at step {step} to {path}")

    def load(self, step):
        path = os.path.join(self.save_dir, f"step_{step}.jax")
        with open(path, "rb") as f:
            flat_params = f.read()
        return jax.tree_util.tree_unflatten(tree_util.tree_structure(flat_params), flat_params)

# 13. مانیتورینگ و لاجینگ
class CosmicMonitor:
    def __init__(self, log_file="training_log.jsonl"):
        self.log_file = log_file
        self.metrics = {"loss": [], "time": [], "step": []}

    def log(self, step, loss, start_time):
        elapsed = time.time() - start_time
        self.metrics["loss"].append(float(loss))
        self.metrics["time"].append(elapsed)
        self.metrics["step"].append(step)
        with open(self.log_file, "a") as f:
            f.write(json.dumps({"step": step, "loss": float(loss), "time": elapsed}) + "\n")
        logger.info(f"Step {step} | Loss: {loss:.4f} | Time: {elapsed:.2f}s")

    def summary(self):
        avg_loss = np.mean(self.metrics["loss"])
        total_time = np.sum(self.metrics["time"])
        logger.info(f"Summary: Avg Loss = {avg_loss:.4f}, Total Time = {total_time:.2f}s")

# 14. آموزش کیهانی
def train_cosmic_dariush(model, tokenizer, mesh, config):
    dataloader = CosmicDataLoader(tokenizer, config.batch_size)
    dataloader.start()
    
    optimizer = optax.adamw(
        learning_rate=optax.cosine_decay_schedule(config.learning_rate, config.total_steps, 0.1),
        weight_decay=0.01,
        b1=0.9, b2=0.95
    )
    
    @hk.transform
    def forward_fn(input_ids, mask=None):
        return model(input_ids, mask)[0]

    def loss_fn(params, batch):
        logits = forward_fn.apply(params, None, batch["input_ids"], batch["mask"])
        labels = batch["labels"]
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        return jnp.mean(loss)

    params = forward_fn.init(jax.random.PRNGKey(42), jnp.ones((1, config.max_seq_len), dtype=jnp.int32))
    opt_state = optimizer.init(params)
    
    @jax.jit
    def update_step(params, opt_state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    checkpoint_mgr = CosmicCheckpointManager()
    monitor = CosmicMonitor()
    start_time = time.time()

    for step, batch in enumerate(tqdm(dataloader, total=config.total_steps, desc="Training Cosmic Dariush")):
        if step >= config.total_steps:
            break
        
        params, opt_state, loss = update_step(params, opt_state, batch)
        monitor.log(step, loss, start_time)
        
        if step % 1000 == 0:
            checkpoint_mgr.save(params, step)
        
        if step % 100 == 0:
            logger.info(f"Step {step} | Loss: {loss:.4f}")

    dataloader.stop()
    monitor.summary()
    checkpoint_mgr.save(params, config.total_steps)
    return params

# 15. اجرا
if __name__ == "__main__":
    tokenizer = CosmicTokenizer()
    tokenizer.train()
    
    mesh_devices = jax.devices()  # فرض چند دستگاه
    mesh = jax.sharding.Mesh(mesh_devices, ("data", "model", "expert"))
    
    with mesh:
        model = GodModeDariushCosmic(config, mesh)
        params = train_cosmic_dariush(model, tokenizer, mesh, config)
        
        # تست تولید متن
        input_text = "جهان از نگاه من یک راز بزرگ است"
        input_ids = tokenizer.batch_encode([input_text])
        generated = model.generate(input_ids, max_len=200)
        print(tokenizer.decode(generated[0]))

# این کد از سورس‌های زیر الهام گرفته شده:
# - DariushGPT (Copyright (c) 2025 hosein davod abadi farahani)
# - xAI Transformer (Copyright 2024 X.AI Corp., Apache License 2.0)
# - الهام از LLaMA, Mixtral, GPT-4, Grok و تکنیک‌های کیهانی 2025
