import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import load_dataset
from tqdm import tqdm

# تنظیمات ساده
class DariushConfig:
    def __init__(self):
        self.vocab_size = 5000  # کوچیک‌تر برای سرعت
        self.emb_size = 256
        self.num_heads = 4
        self.hidden_size = 512
        self.max_seq_len = 128
        self.batch_size = 8
        self.learning_rate = 2e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = DariushConfig()

# توکنایزر
class PersianTokenizer:
    def __init__(self):
        self.tokenizer = Tokenizer(models.BPE())
        
    def train(self):
        dataset = load_dataset("oscar", "unshuffled_deduplicated_fa", split="train[:1%]")
        trainer = trainers.BpeTrainer(vocab_size=config.vocab_size, special_tokens=["[PAD]", "[BOS]", "[EOS]"])
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        self.tokenizer.train_from_iterator(dataset["text"], trainer=trainer)
        self.tokenizer.save("tokenizer.json")
        
    def encode(self, text):
        return self.tokenizer.encode(text).ids
    
    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

# دیتاست
class PersianDataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.texts = load_dataset("oscar", "unshuffled_deduplicated_fa", split="train[:1%]")["text"][:100]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(self.texts[idx])[:config.max_seq_len]
        input_ids = tokens[:-1]
        labels = tokens[1:]
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels)}

# مدل
class DariushGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.emb_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config.emb_size, nhead=config.num_heads), num_layers=2
        )
        self.output = nn.Linear(config.emb_size, config.vocab_size)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        return self.output(x)

# آموزش
tokenizer = PersianTokenizer()
tokenizer.train()
dataset = PersianDataset(tokenizer)
loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
model = DariushGPT(config).to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

for epoch in range(3):  # 3 epoch برای تست
    for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
        input_ids = batch["input_ids"].to(config.device)
        labels = batch["labels"].to(config.device)
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = nn.CrossEntropyLoss()(outputs.view(-1, config.vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Loss: {loss.item()}")

# تولید متن
def generate_text(model, tokenizer, prompt):
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(config.device)
    for _ in range(20):
        outputs = model(input_ids)
        next_token = torch.argmax(outputs[:, -1, :], dim=-1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
    return tokenizer.decode(input_ids[0].tolist())

print(generate_text(model, tokenizer, "شب تاریک بود و"))
