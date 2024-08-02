import json
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import wandb
import os

class JSONLDataset(Dataset):
    def __init__(self, file_path, tokenizer, intent_to_label=None):
        self.data = []
        self.tokenizer = tokenizer
        self.intent_to_label = intent_to_label or {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)
                if item['intent'] not in self.intent_to_label:
                    self.intent_to_label[item['intent']] = len(self.intent_to_label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['utt']
        intent = item['intent']
        
        encoding = self.tokenizer(text, 
                                  truncation=True, 
                                  padding='max_length', 
                                  max_length=512, 
                                  return_tensors='pt')
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.intent_to_label[intent], dtype=torch.long)
        }

    def get_intent_to_label(self):
        return self.intent_to_label

def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1)
        predictions.extend(pred.cpu().numpy())
        true_labels.extend(batch["labels"].cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main():

    wandb.init(
        project="bert-finetuning",
        name="multilingual-intent-classification-lora",
        config={
            "model": "bert-base-multilingual-cased",
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "learning_rate": 2e-5,
            "batch_size": 16,
            "num_epochs": 3
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    data_dir = '/home/sarim.hashmi/Downloads/NLP-702_assignment/dataset/1.0/data'
    train_files = [f for f in os.listdir(data_dir) if f.endswith('.jsonl')]

    intent_to_label = {}
    train_datasets = []
    for f in train_files:
        dataset = JSONLDataset(os.path.join(data_dir, f), tokenizer, intent_to_label)
        train_datasets.append(dataset)
        intent_to_label.update(dataset.get_intent_to_label())

    train_dataset = ConcatDataset(train_datasets)
    num_labels = len(intent_to_label)


    wandb.log({"num_intents": num_labels})


    model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=num_labels)


    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=wandb.config.lora_r,
        lora_alpha=wandb.config.lora_alpha,
        lora_dropout=wandb.config.lora_dropout,
        bias="none",
        target_modules=["query", "key", "value"]
    )


    model = get_peft_model(model, peft_config)
    

    wandb.watch(model, log="all", log_freq=100)


    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")
    wandb.log({
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_percent": 100 * trainable_params / total_params
    })

    model.to(device)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=wandb.config.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=wandb.config.batch_size)


    wandb.log({
        "train_size": len(train_dataset),
        "val_size": len(val_dataset)
    })

    before_metrics = evaluate(model, val_dataloader, device)
    wandb.log({"before_" + k: v for k, v in before_metrics.items()})

    optimizer = AdamW(model.parameters(), lr=wandb.config.learning_rate)
    num_epochs = wandb.config.num_epochs

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        avg_train_loss = total_loss / len(train_dataloader)
        wandb.log({"epoch": epoch+1, "train_loss": avg_train_loss})
        
        metrics = evaluate(model, val_dataloader, device)
        wandb.log({f"epoch_{epoch+1}_" + k: v for k, v in metrics.items()})

    after_metrics = evaluate(model, val_dataloader, device)
    wandb.log({"after_" + k: v for k, v in after_metrics.items()})

    # Save the LoRA model
    model.save_pretrained("./lora_bert_intent_classification")

    wandb.finish()

if __name__ == "__main__":
    main()