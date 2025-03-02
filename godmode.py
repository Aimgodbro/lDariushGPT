# GodModeDariushCosmic: The ultimate cosmic transformer, fully expanded to over 1000 lines
# Copyright (c) 2025 hosein davod abadi farahani & cosmic enhancements

import jax
import jax.numpy as jnp
import haiku as hk
from jax import config as jax_config
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax.lax import with_sharding_constraint as pjit_sharding_constraint
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import load_dataset, Dataset
from typing import Optional, List, Dict, Any, Tuple, Callable
from tqdm import tqdm
import functools
import logging
import optax
import numpy as np
from dataclasses import dataclass, field
import jax.tree_util as tree_util
import threading
import queue
import os
import time
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from jax.experimental import mesh_utils
import multiprocessing as mp
from collections import deque
import hashlib
import shutil

jax_config.update("jax_spmd_mode", "allow_all")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 1. تنظیمات کیهانی پیشرفته
@dataclass
class CosmicConfig:
    vocab_size: int = 131072
    emb_size: int = 8192
    num_q_heads: int = 128
    num_kv_heads: int = 16
    key_size: int = 128
    num_layers: int = 64
    num_experts: int = 64
    num_selected_experts: int = 8
    widening_factor: float = 4.5
    max_seq_len: int = 16384
    init_scale: float = 0.01
    dropout_rate: float = 0.1
    sparse_factor: int = 4
    data_axis: str = "data"
    model_axis: str = "model"
    expert_axis: str = "expert"
    shard_activations: bool = True
    use_swiglu: bool = True
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    batch_size: int = 32
    num_micro_batches: int = 4
    learning_rate: float = 6e-5
    warmup_steps: int = 2000
    total_steps: int = 100000
    checkpoint_interval: int = 1000
    log_interval: int = 100
    cache_size: int = 10000
    num_workers: int = 8
    prefetch_size: int = 20
    special_tokens: Dict[str, int] = field(default_factory=lambda: {
        "[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3, "[CLS]": 4, "[SEP]": 5
    })

    def partition_rules(self) -> List[Tuple[Tuple[str, ...], P]]:
        return [
            (("embedding", "w"), P(None, "data", "model")),
            (("multi_head_attention", "(query|key|value)", "w"), P("data", "model")),
            (("multi_head_attention", "linear", "w"), P("model", "data")),
            (("moe", "router", "w"), P("data", "expert")),
            (("moe", "expert", "w"), P("expert", "data", "model")),
            (("moe", "expert_out", "w"), P("expert", "model", "data")),
            (("rms_norm", "scale"), P(None)),
            (("output", "w"), P("model", "data")),
            (("kv_cache", "k"), P("data", "model")),
            (("kv_cache", "v"), P("data", "model")),
        ]

    def get_mesh(self) -> jax.sharding.Mesh:
        devices = jax.devices()
        return jax.sharding.Mesh(devices, ("data", "model", "expert"))

    def validate(self):
        assert self.num_q_heads % self.num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads"
        assert self.max_seq_len > 0, "max_seq_len must be positive"
        logger.info("Configuration validated successfully.")

config = CosmicConfig()

# 2. توکنایزر کیهانی پیشرفته
class CosmicTokenizer:
    def __init__(self):
        self.tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        self.special_tokens = config.special_tokens
        self.cache = {}
        self.cache_size = config.cache_size
        self.cache_hits = 0
        self.cache_misses = 0

    def train(self, data_path: str = "oscar_fa", cache_file: str = "tokenizer_cache.pkl"):
        logger.info("Initializing tokenizer training...")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                self.tokenizer = pickle.load(f)
            logger.info("Loaded tokenizer from cache.")
            return

        dataset = load_dataset("oscar", "unshuffled_deduplicated_fa", split="train[:20%]")
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=config.vocab_size,
            special_tokens=list(self.special_tokens.keys()),
            min_frequency=2,
            show_progress=True,
            continuing_subword_prefix="##"
        )
        self.tokenizer.train_from_iterator(dataset["text"], trainer=trainer)
        self.tokenizer.save("cosmic_dariush_tokenizer.json")
        with open(cache_file, "wb") as f:
            pickle.dump(self.tokenizer, f)
        logger.info("Tokenizer trained and saved to cache.")

    def encode(self, text: str) -> jnp.ndarray:
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.cache:
            self.cache_hits += 1
            return self.cache[text_hash]
        tokens = jnp.array(self.tokenizer.encode(text).ids)
        if len(self.cache) < self.cache_size:
            self.cache[text_hash] = tokens
        else:
            self.cache.pop(next(iter(self.cache)))
            self.cache[text_hash] = tokens
        self.cache_misses += 1
        return tokens

    def decode(self, tokens: jnp.ndarray) -> str:
        return self.tokenizer.decode(tokens.tolist())

    def pad(self, sequences: List[List[int]], max_len: int) -> jnp.ndarray:
        padded = []
        for seq in sequences:
            seq = seq[:max_len]
            padded_seq = seq + [self.special_tokens["[PAD]"]] * (max_len - len(seq))
            padded.append(padded_seq)
        return jnp.array(padded)

    def batch_encode(self, texts: List[str], max_len: int = config.max_seq_len) -> Tuple[jnp.ndarray, jnp.ndarray]:
        encoded = []
        for text in texts:
            tokens = self.encode(text)
            encoded.append(tokens.tolist())
        input_ids = self.pad(encoded, max_len)
        mask = (input_ids != self.special_tokens["[PAD]"])[:, None, None, :]
        return input_ids, mask

    def encode_parallel(self, texts: List[str], num_threads: int = config.num_workers) -> Tuple[jnp.ndarray, jnp.ndarray]:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            encoded = list(executor.map(self.encode, texts))
        return self.batch_encode([e.tolist() for e in encoded])

    def clear_cache(self):
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Tokenizer cache cleared.")

    def cache_stats(self) -> Dict[str, int]:
        return {"hits": self.cache_hits, "misses": self.cache_misses, "size": len(self.cache)}

# 3. دیتالودر کیهانی پیشرفته
class CosmicDataLoader:
    def __init__(self, tokenizer: CosmicTokenizer, batch_size: int, num_workers: int = config.num_workers, prefetch_size: int = config.prefetch_size):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_size = prefetch_size
        self.dataset = load_dataset("oscar", "unshuffled_deduplicated_fa", split="train[:20%]")["text"]
        self.total_samples = len(self.dataset)
        self.queue = queue.Queue(maxsize=prefetch_size)
        self.running = False
        self.cache = deque(maxlen=1000)
        self.cache_lock = threading.Lock()

    def start(self):
        self.running = True
        self.workers = []
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker_fn, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
        logger.info(f"Started {self.num_workers} data loader workers.")

    def stop(self):
        self.running = False
        for worker in self.workers:
            worker.join()
        logger.info("Data loader workers stopped.")

    def _worker_fn(self, worker_id: int):
        while self.running:
            try:
                with self.cache_lock:
                    if self.cache and np.random.random() < 0.3:  # 30% chance to use cache
                        batch = self.cache[np.random.randint(len(self.cache))]
                    else:
                        start_idx = np.random.randint(0, self.total_samples - self.batch_size)
                        batch_texts = self.dataset[start_idx:start_idx + self.batch_size]
                        input_ids, mask = self.tokenizer.batch_encode(batch_texts)
                        batch = {
                            "input_ids": input_ids,
                            "labels": input_ids,
                            "mask": mask
                        }
                        self.cache.append(batch)
                self.queue.put(batch, timeout=10)
            except queue.Full:
                if not self.running:
                    break
                time.sleep(1)
            except Exception as e:
                logger.error(f"Worker {worker_id} encountered error: {e}")

    def __iter__(self):
        return self

    def __next__(self):
        if not self.running:
            raise StopIteration
        return self.queue.get()

    def prefetch(self):
        logger.info("Starting data prefetching...")
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._worker_fn, i) for i in range(self.num_workers)]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Prefetching error: {e}")

    def stats(self) -> Dict[str, Any]:
        return {
            "queue_size": self.queue.qsize(),
            "cache_size": len(self.cache),
            "total_samples": self.total_samples
        }

# 4. RMSNorm کیهانی
class CosmicRMSNorm(hk.Module):
    def __init__(self, emb_size: int, eps: float = 1e-6, name: str = "rms_norm"):
        super().__init__(name=name)
        self.emb_size = emb_size
        self.eps = eps

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = hk.get_parameter("scale", [self.emb_size], init=jnp.ones)
        scale = pjit_sharding_constraint(scale, P(None))
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(variance + self.eps) * scale
        return normed.astype(jnp.bfloat16)

    def reset(self):
        hk.set_parameter("scale", jnp.ones(self.emb_size))

# 5. Rotary Embedding کیهانی
class CosmicRotaryEmbedding(hk.Module):
    def __init__(self, dim: int, base: int = 10000, max_seq_len: int = config.max_seq_len, name: str = "rotary_emb"):
        super().__init__(name=name)
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def __call__(self, x: jnp.ndarray, offset: int = 0) -> jnp.ndarray:
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
    def __init__(self, hidden_size: int, name: str = "swiglu"):
        super().__init__(name=name)
        self.hidden_size = hidden_size

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        w1 = hk.Linear(self.hidden_size, name="w1", w_init=hk.initializers.TruncatedNormal(stddev=0.02))
        w2 = hk.Linear(self.hidden_size, name="w2", w_init=hk.initializers.TruncatedNormal(stddev=0.02))
        return jax.nn.silu(w1(x)) * w2(x)

# 7. Flash Attention کیهانی
class CosmicFlashAttention(hk.Module):
    def __init__(self, num_heads: int, key_size: int, block_size: int = 128, name: str = "flash_attention"):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.block_size = block_size

    def __call__(self, q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
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

        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        q_blocks = q.reshape(batch, num_blocks, self.block_size, self.num_heads, self.key_size)
        k_blocks = k.reshape(batch, num_blocks, self.block_size, self.num_heads, self.key_size)
        v_blocks = v.reshape(batch, num_blocks, self.block_size, self.num_heads, self.key_size)
        mask_blocks = mask.reshape(batch, 1, num_blocks, self.block_size) if mask is not None else None

        @functools.partial(shard_map, mesh=config.get_mesh(), in_specs=(P("data", None, "model"), P("data", None, "model"), P("data", None, "model"), P("data", None)),
                           out_specs=P("data", "model"), check_rep=False)
        def sharded_block_attention(qb, kb, vb, mb):
            return block_attention(qb, kb, vb, mb)

        outputs = jax.vmap(sharded_block_attention)(q_blocks, k_blocks, v_blocks, mask_blocks)
        return outputs.reshape(batch, seq_len, self.num_heads * self.key_size)

# 8. Sparse Attention کیهانی
class CosmicSparseAttention(hk.Module):
    def __init__(self, num_heads: int, key_size: int, sparse_factor: int = config.sparse_factor, name: str = "sparse_attention"):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.sparse_factor = sparse_factor

    def __call__(self, q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        batch, seq_len, _ = q.shape
        q = q.reshape(batch, seq_len, self.num_heads, self.key_size)
        k = k.reshape(batch, seq_len, self.num_heads, self.key_size)
        v = v.reshape(batch, seq_len, self.num_heads, self.key_size)

        sparse_seq_len = seq_len // self.sparse_factor
        q_sparse = q[:, ::self.sparse_factor, :, :]
        k_sparse = k[:, ::self.sparse_factor, :, :]
        v_sparse = v[:, ::self.sparse_factor, :, :]

        attn_logits = jnp.einsum("...qhd,...khd->...hqk", q_sparse, k_sparse) / jnp.sqrt(self.key_size)
        if mask is not None:
            mask_sparse = mask[:, :, ::self.sparse_factor]
            attn_logits = jnp.where(mask_sparse, attn_logits, -1e30)
        attn_weights = jax.nn.softmax(attn_logits)
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, v_sparse)
        return attn_output.reshape(batch, sparse_seq_len, self.num_heads * self.key_size)

# 9. MoE کیهانی
class CosmicRouter(hk.Module):
    def __init__(self, num_experts: int, num_selected_experts: int, name: str = "router"):
        super().__init__(name=name)
        self.num_experts = num_experts
        self.num_selected_experts = num_selected_experts

    def __call__(self, inputs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        w = hk.get_parameter("w", [inputs.shape[-1], self.num_experts], init=hk.initializers.TruncatedNormal(stddev=0.02))
        w = pjit_sharding_constraint(w, P("data", "expert"))
        logits = jnp.dot(inputs.astype(jnp.float32), w)
        noise = jax.random.gumbel(jax.random.PRNGKey(0), logits.shape) * 0.05
        probs = jax.nn.softmax(logits + noise)
        gates, indices = jax.lax.top_k(probs, self.num_selected_experts)
        return gates, indices

class CosmicMoELayer(hk.Module):
    def __init__(self, config: CosmicConfig, mesh: jax.sharding.Mesh, name: str = "moe"):
        super().__init__(name=name)
        self.config = config
        self.mesh = mesh
        self.router = CosmicRouter(config.num_experts, config.num_selected_experts)

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        gates, indices = self.router(inputs)
        expert_outputs = []

        def expert_fn(x):
            w = hk.Linear(int(self.config.widening_factor * self.config.emb_size), name="expert",
                         w_init=hk.initializers.TruncatedNormal(stddev=0.02))
            w_out = hk.Linear(self.config.emb_size, name="expert_out",
                            w_init=hk.initializers.TruncatedNormal(stddev=0.02))
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

# 10. توجه چندسر کیهانی
class CosmicMultiHeadAttention(hk.Module):
    def __init__(self, config: CosmicConfig, name: str = "multi_head_attention"):
        super().__init__(name=name)
        self.config = config
        self.rotary = CosmicRotaryEmbedding(config.key_size)

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, kv_cache: Optional[Dict] = None) -> Tuple[jnp.ndarray, Dict]:
        q_w = hk.Linear(self.config.num_q_heads * self.config.key_size, name="query",
                       w_init=hk.initializers.TruncatedNormal(stddev=0.02))
        k_w = hk.Linear(self.config.num_kv_heads * self.config.key_size, name="key",
                       w_init=hk.initializers.TruncatedNormal(stddev=0.02))
        v_w = hk.Linear(self.config.num_kv_heads * self.config.key_size, name="value",
                       w_init=hk.initializers.TruncatedNormal(stddev=0.02))
        out_w = hk.Linear(self.config.emb_size, name="linear",
                         w_init=hk.initializers.TruncatedNormal(stddev=0.02))

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
            sparse_attn = CosmicSparseAttention(self.config.num_q_heads, self.config.key_size, self.config.sparse_factor)
            attn_output = sparse_attn(q, k, v, mask)

        return out_w(attn_output), {"k": k, "v": v}

# 11. لایه کیهانی
class CosmicDariushLayer(hk.Module):
    def __init__(self, config: CosmicConfig, mesh: jax.sharding.Mesh, layer_idx: int, name: str = "cosmic_layer"):
        super().__init__(name=f"{name}_{layer_idx}")
        self.config = config
        self.mesh = mesh
        self.layer_idx = layer_idx
        self.attn = CosmicMultiHeadAttention(config)
        self.moe = CosmicMoELayer(config, mesh)
        self.norm1 = CosmicRMSNorm(config.emb_size)
        self.norm2 = CosmicRMSNorm(config.emb_size)
        self.dropout = hk.dropout if config.dropout_rate > 0 else lambda x: x

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, kv_cache: Optional[Dict] = None) -> Tuple[jnp.ndarray, Dict]:
        if self.config.gradient_checkpointing:
            attn_out, new_cache = hk.checkpoint(lambda x: self.attn(self.norm1(x), mask, kv_cache))(x)
        else:
            attn_out, new_cache = self.attn(self.norm1(x), mask, kv_cache)
        x = x + self.dropout(attn_out, rate=self.config.dropout_rate, salt=jax.random.PRNGKey(self.layer_idx))
        moe_out = self.moe(self.norm2(x))
        x = x + self.dropout(moe_out, rate=self.config.dropout_rate, salt=jax.random.PRNGKey(self.layer_idx + 1))
        return x, new_cache

# 12. مدل اصلی کیهانی
class GodModeDariushCosmic(hk.Module):
    def __init__(self, config: CosmicConfig, mesh: jax.sharding.Mesh, name: str = "godmode_dariush_cosmic"):
        super().__init__(name=name)
        self.config = config
        self.mesh = mesh
        self.embedding = hk.Embed(config.vocab_size, config.emb_size, name="embedding",
                                w_init=hk.initializers.TruncatedNormal(stddev=config.init_scale))
        self.layers = [CosmicDariushLayer(config, mesh, i) for i in range(config.num_layers)]
        self.norm = CosmicRMSNorm(config.emb_size)
        self.output = hk.Linear(config.vocab_size, name="output",
                              w_init=hk.initializers.TruncatedNormal(stddev=config.init_scale))

    def __call__(self, input_ids: jnp.ndarray, mask: Optional[jnp.ndarray] = None, kv_cache: Optional[List[Dict]] = None) -> Tuple[jnp.ndarray, List[Dict]]:
        x = self.embedding(input_ids)
        x = pjit_sharding_constraint(x, P(self.config.data_axis, None, self.config.model_axis))
        new_kv_cache = [] if kv_cache is None else kv_cache

        for i, layer in enumerate(self.layers):
            x, layer_cache = layer(x, mask, new_kv_cache[i] if kv_cache else None)
            new_kv_cache.append(layer_cache)

        x = self.norm(x)
        logits = self.output(x)
        return logits, new_kv_cache

    def init_memory(self, batch_size: int, seq_len: int) -> List[Dict]:
        return [{"k": jnp.zeros((batch_size, seq_len, self.config.num_kv_heads, self.config.key_size), dtype=jnp.bfloat16),
                 "v": jnp.zeros((batch_size, seq_len, self.config.num_kv_heads, self.config.key_size), dtype=jnp.bfloat16)}
                for _ in range(self.config.num_layers)]

    def generate(self, input_ids: jnp.ndarray, max_len: int = 200, temperature: float = 0.7, top_k: int = 40, top_p: float = 0.9, beam_width: int = 5) -> jnp.ndarray:
        kv_cache = self.init_memory(input_ids.shape[0], input_ids.shape[1])
        beams = [(input_ids, 0.0, kv_cache)]  # (sequence, score, cache)

        for step in range(max_len):
            new_beams = []
            for seq, score, cache in beams:
                logits, new_cache = self(seq, kv_cache=cache)
                next_logits = logits[:, -1, :] / temperature
                top_k_logits, top_k_tokens = jax.lax.top_k(next_logits, top_k)
                probs = jax.nn.softmax(top_k_logits)
                
                # Top-p filtering (nucleus sampling)
                sorted_probs = jnp.sort(probs, axis=-1, descending=True)
                cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)
                mask = cumulative_probs <= top_p
                filtered_probs = jnp.where(mask, probs, 0.0)
                filtered_probs /= jnp.sum(filtered_probs, axis=-1, keepdims=True)
                
                for i in range(top_k):
                    if filtered_probs[:, i] > 0:
                        new_seq = jnp.concatenate([seq, top_k_tokens[:, i:i+1]], axis=1)
                        new_score = score + jnp.log(filtered_probs[:, i])
                        new_beams.append((new_seq, new_score, new_cache))

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            if jnp.all(beams[0][0][:, -1] == self.config.special_tokens["[EOS]"]):
                break

        return beams[0][0]

    def evaluate(self, input_ids: jnp.ndarray, labels: jnp.ndarray) -> float:
        logits, _ = self(input_ids)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        return jnp.mean(loss)

# 13. مدیریت چک‌پوینت کیهانی
class CosmicCheckpointManager:
    def __init__(self, save_dir: str = "cosmic_checkpoints", max_checkpoints: int = 5):
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        os.makedirs(save_dir, exist_ok=True)
        self.checkpoints = []
        self.lock = threading.Lock()

    def save(self, params: Any, step: int, metadata: Dict = None):
        with self.lock:
            path = os.path.join(self.save_dir, f"checkpoint_step_{step}.jax")
            flat_params, tree_def = jax.tree_util.tree_flatten(params)
            data = {"params": flat_params, "tree_def": tree_def, "metadata": metadata or {}}
            with open(path, "wb") as f:
                pickle.dump(data, f)
            
            self.checkpoints.append((step, path))
            if len(self.checkpoints) > self.max_checkpoints:
                old_step, old_path = self.checkpoints.pop(0)
                os.remove(old_path)
                logger.info(f"Removed old checkpoint: {old_path}")
            
            logger.info(f"Saved checkpoint at step {step} to {path}")

    def load(self, step: int) -> Tuple[Any, Dict]:
        path = os.path.join(self.save_dir, f"checkpoint_step_{step}.jax")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        params = jax.tree_util.tree_unflatten(data["tree_def"], data["params"])
        return params, data["metadata"]

    def latest_checkpoint(self) -> Optional[int]:
        if not self.checkpoints:
            return None
        return max(self.checkpoints, key=lambda x: x[0])[0]

    def cleanup(self):
        with self.lock:
            for _, path in self.checkpoints:
                if os.path.exists(path):
                    os.remove(path)
            self.checkpoints.clear()
            logger.info("All checkpoints cleaned up.")

# 14. مانیتورینگ کیهانی
class CosmicMonitor:
    def __init__(self, log_file: str = "cosmic_training_log.jsonl", plot_dir: str = "cosmic_plots"):
        self.log_file = log_file
        self.plot_dir = plot_dir
        os.makedirs(plot_dir, exist_ok=True)
        self.metrics = {"step": [], "loss": [], "time": [], "lr": [], "grad_norm": []}
        self.lock = threading.Lock()

    def log(self, step: int, loss: float, lr: float, grad_norm: float, start_time: float):
        with self.lock:
            elapsed = time.time() - start_time
            self.metrics["step"].append(step)
            self.metrics["loss"].append(float(loss))
            self.metrics["time"].append(elapsed)
            self.metrics["lr"].append(float(lr))
            self.metrics["grad_norm"].append(float(grad_norm))
            
            log_entry = {"step": step, "loss": float(loss), "time": elapsed, "lr": float(lr), "grad_norm": float(grad_norm)}
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            logger.info(f"Step {step} | Loss: {loss:.4f} | LR: {lr:.6f} | Grad Norm: {grad_norm:.4f} | Time: {elapsed:.2f}s")

    def plot(self, metric: str):
        plt.figure(figsize=(12, 8))
        plt.plot(self.metrics["step"], self.metrics[metric], label=metric.capitalize())
        plt.xlabel("Step")
        plt.ylabel(metric.capitalize())
        plt.title(f"{metric.capitalize()} over Training Steps")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plot_dir, f"{metric}_plot.png"))
        plt.close()
        logger.info(f"Plotted {metric} to {self.plot_dir}")

    def summary(self):
        with self.lock:
            avg_loss = np.mean(self.metrics["loss"])
            total_time = np.sum(self.metrics["time"])
            max_grad_norm = np.max(self.metrics["grad_norm"])
            logger.info(f"Training Summary: Avg Loss = {avg_loss:.4f}, Total Time = {total_time:.2f}s, Max Grad Norm = {max_grad_norm:.4f}")
            for metric in ["loss", "lr", "grad_norm"]:
                self.plot(metric)

    def export(self, export_file: str = "metrics_summary.json"):
        with self.lock:
            with open(export_file, "w") as f:
                json.dump(self.metrics, f)
            logger.info(f"Metrics exported to {export_file}")

# 15. بهینه‌ساز کیهانی
def cosmic_optimizer(config: CosmicConfig) -> optax.GradientTransformation:
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.total_steps - config.warmup_steps,
        end_value=config.learning_rate * 0.1
    )
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, b1=0.9, b2=0.95, weight_decay=0.01),
        optax.scale_by_schedule(lambda step: 1.0)
    )

# 16. آموزش کیهانی
def train_cosmic_dariush(model: GodModeDariushCosmic, tokenizer: CosmicTokenizer, mesh: jax.sharding.Mesh, config: CosmicConfig):
    dataloader = CosmicDataLoader(tokenizer, config.batch_size)
    dataloader.start()
    
    optimizer = cosmic_optimizer(config)
    
    @hk.transform
    def forward_fn(input_ids: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        return model(input_ids, mask)[0]

    def loss_fn(params: Any, batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        logits = forward_fn.apply(params, None, batch["input_ids"], batch["mask"])
        labels = batch["labels"]
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        return jnp.mean(loss)

    params = forward_fn.init(jax.random.PRNGKey(42), jnp.ones((1, config.max_seq_len), dtype=jnp.int32))
    opt_state = optimizer.init(params)
    
    @jax.jit
    def update_step(params: Any, opt_state: Any, batch: Dict[str, jnp.ndarray]) -> Tuple[Any, Any, jnp.ndarray, float]:
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        grad_norm = optax.global_norm(grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, grad_norm

    checkpoint_mgr = CosmicCheckpointManager()
    monitor = CosmicMonitor()
    start_time = time.time()
    step = 0

    latest_step = checkpoint_mgr.latest_checkpoint()
    if latest_step is not None:
        params, metadata = checkpoint_mgr.load(latest_step)
        step = latest_step + 1
        logger.info(f"Resumed training from step {step} with metadata: {metadata}")

    for batch in tqdm(dataloader, total=config.total_steps, desc="Training Cosmic Dariush"):
        if step >= config.total_steps:
            break
        
        micro_batches = [dict(tree_util.tree_map(lambda x: x[i*self.batch_size//self.num_micro_batches:(i+1)*self.batch_size//self.num_micro_batches], batch)) 
                         for i in range(config.num_micro_batches)]
        accumulated_grads = None
        
        for micro_batch in micro_batches:
            params, opt_state, micro_loss, micro_grad_norm = update_step(params, opt_state, micro_batch)
            if accumulated_grads is None:
                accumulated_grads = micro_grad_norm
            else:
                accumulated_grads += micro_grad_norm
        
        grad_norm = accumulated_grads / config.num_micro_batches
        if step % config.log_interval == 0:
            monitor.log(step, micro_loss, config.learning_rate, grad_norm, start_time)
        
        if step % config.checkpoint_interval == 0 and step > 0:
            checkpoint_mgr.save(params, step, {"loss": float(micro_loss), "step": step})
        
        step += 1

    dataloader.stop()
    monitor.summary()
    monitor.export()
    checkpoint_mgr.save(params, config.total_steps, {"final_step": step})
    return params

# 17. تست و اعتبارسنجی
def validate_cosmic_dariush(model: GodModeDariushCosmic, tokenizer: CosmicTokenizer, test_texts: List[str]) -> float:
    input_ids, mask = tokenizer.batch_encode(test_texts)
    labels = input_ids
    loss = model.evaluate(input_ids, labels)
    logger.info(f"Validation Loss: {loss:.4f}")
    return loss

def generate_samples(model: GodModeDariushCosmic, tokenizer: CosmicTokenizer, prompts: List[str], num_samples: int = 5) -> List[str]:
    samples = []
    for prompt in prompts[:num_samples]:
        input_ids = tokenizer.batch_encode([prompt])
        generated = model.generate(input_ids, max_len=200)
        decoded = tokenizer.decode(generated[0])
        samples.append(decoded)
        logger.info(f"Generated: {decoded}")
    return samples

# 18. اجرا
if __name__ == "__main__":
    tokenizer = CosmicTokenizer()
    tokenizer.train()
    
    mesh = config.get_mesh()
    with mesh:
        model = GodModeDariushCosmic(config, mesh)
        params = train_cosmic_dariush(model, tokenizer, mesh, config)
        
        # تست اعتبارسنجی
        test_texts = [
            "جهان از نگاه من یک راز بزرگ است",
            "زندگی پر از شگفتی است",
            "آینده در دستان ماست",
            "علم کلید پیشرفت است",
            "هنر زبان احساسات است"
        ]
        validate_cosmic_dariush(model, tokenizer, test_texts)
        
        # تولید نمونه
        samples = generate_samples(model, tokenizer, test_texts)
        for i, sample in enumerate(samples):
            print(f"Sample {i+1}: {sample}")

# این کد از سورس‌های زیر الهام گرفته شده:
# - DariushGPT (Copyright (c) 2025 hosein davod abadi farahani)
# - xAI Transformer (Copyright 2024 X.AI Corp., Apache License 2.0)
# - الهام از LLaMA, Mixtral, GPT-4, Grok و تکنیک‌های کیهانی 2025
