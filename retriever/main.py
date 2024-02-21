from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader, random_split
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, Features, Value, ClassLabel
from transformers import BertModel
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import json

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "query": item["query"],
            "positive": item["positive"],
            "negatives": item["negatives"]
        }
    
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

# batch_processing
def process_batch(model, batch, tokenizer):
    # Tokenize the inputs
    inputs_query = tokenizer(batch['query'], padding=True, truncation=True, return_tensors="pt")
    inputs_positive = tokenizer(batch['positive'], padding=True, truncation=True, return_tensors="pt")

    # Tokenize each negative sample individually
    negative_embs = []
    for negative in batch['negatives']:
        inputs_negative = tokenizer(negative, padding=True, truncation=True, return_tensors="pt")
        inputs_negative = {k: v.to(model.device) for k, v in inputs_negative.items()}
        negative_emb = model(**inputs_negative).last_hidden_state[:, 0, :]
        negative_embs.append(negative_emb)

    # Move to the same device as the model
    inputs_query = {k: v.to(model.device) for k, v in inputs_query.items()}
    inputs_positive = {k: v.to(model.device) for k, v in inputs_positive.items()}

    # Get the embeddings from the model
    anchor_emb = model(**inputs_query).last_hidden_state[:, 0, :]
    positive_emb = model(**inputs_positive).last_hidden_state[:, 0, :]

    # Aggregate negative embeddings
    negative_emb = torch.stack(negative_embs).mean(dim=0)

    return anchor_emb, positive_emb, negative_emb

if __name__ == '__main__':
    # load dataset
    file_path = 'pubmed_gene_data.json'
    with open(file_path, 'r') as file:
        dataset = json.load(file)

    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)  
    val_size = dataset_size - train_size  

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)  # Usually, we don't need to shuffle the validation data 

    # Load the model
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")

    # Define the optimizer with weight decay (L2 regularization)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01) 

    loss_fn = TripletLoss()

    # Early Stopping Parameters
    best_val_loss = float('inf')
    epochs_no_improve = 0
    n_epochs_stop = 3  # Number of epochs to wait for improvement before stopping

    # Custom Training Loop
    for epoch in tqdm(range(10)):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader):
            # Process your batch to get anchor, positive, and negative embeddings
            anchor_emb, positive_emb, negative_emb = process_batch(model, batch, tokenizer)

            # Compute the loss
            loss = loss_fn(anchor_emb, positive_emb, negative_emb)
            print(loss)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch}: Average Training Loss = {avg_train_loss}")

        # Validation step for early stopping
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for val_batch in val_dataloader:  # Assuming you have a validation dataloader

                avg_val_loss = total_val_loss / len(val_dataloader)
                print(f"Epoch {epoch}: Average Validation Loss = {avg_val_loss}")

                # Check for early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    # Save the best model
                    torch.save(model, 'finetuned_bge_large.bin')
                    torch.save(model.state_dict(), 'retriever_weight')
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve == n_epochs_stop:
                        print("Early stopping triggered")
                        break

