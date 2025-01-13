import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from transformers import BertTokenizer, BertModel
from torch_geometric.data import Data, DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader
import json
from sklearn.model_selection import train_test_split
from torch.optim import Adam
import pickle
from torch.nn.utils.rnn import pad_sequence

# Load label definitions from JSON file
def load_labels(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    labels = data[0]['labels']
    groups = {g['id']: g['name'] for g in data[0]['groups']}
    return labels, groups

# Custom Dataset for Attribute Classification
class AttributeDataset:
    def __init__(self, labels, tokenizer, groups):
        self.tokenizer = tokenizer
        self.groups = groups
        self.samples = []
        self.labels = []

        for label in labels:
            group_id = label['group_id']
            for sample in label.get('samples', []):
                self.samples.append(sample)
                self.labels.append(group_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]
        tokens = self.tokenizer(sample, padding='max_length', truncation=True, return_tensors="pt")
        return tokens, label

# BERT + GNN Model
class BertGNNClassifier(nn.Module):
    def __init__(self, gnn_hidden_dim, num_classes):
        super(BertGNNClassifier, self).__init__()
        self.gnn1 = GCNConv(768, gnn_hidden_dim)  # Input size = 768 (BERT embedding dim)
        self.gnn2 = GCNConv(gnn_hidden_dim, gnn_hidden_dim)
        self.fc = nn.Linear(gnn_hidden_dim, num_classes)
    
    def forward(self, x, edge_index):
        # GNN Layers
        x = torch.relu(self.gnn1(x, edge_index))
        x = torch.relu(self.gnn2(x, edge_index))
        
        # Classification
        return self.fc(x)


# Training Function
def train_model(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in data_loader:
        x = batch.x.to(device)  # Node features (precomputed embeddings)
        edge_index = batch.edge_index.to(device)  # Graph edges
        labels = batch.y.to(device)  # Graph labels

        optimizer.zero_grad()

        # Forward pass
        outputs = model(x, edge_index)  # Pass node features and edges to the model
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)

# Evaluation Function
def evaluate_model(model, data_loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            labels = batch.y.to(device)

            outputs = model(x, edge_index)
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    return total_correct / total_samples


def compute_bert_embeddings(samples, tokenizer, bert_model, device):
    embeddings = []
    bert_model = bert_model.to(device)
    bert_model.eval()

    with torch.no_grad():
        for sample in samples:
            tokens = tokenizer(sample, padding="max_length", truncation=True, return_tensors="pt")
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            embeddings.append(cls_embedding.cpu())

    return torch.cat(embeddings, dim=0)

def construct_graph(samples, labels, embeddings, max_nodes):
    edge_index = []
    labels_tensor = torch.tensor(labels[:max_nodes], dtype=torch.long)

    for idx in range(len(samples[:max_nodes])):
        if idx > 0:  # Connect sequential nodes (example)
            edge_index.append([idx - 1, idx])
            edge_index.append([idx, idx - 1])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = embeddings[:max_nodes]  # Use precomputed embeddings

    return Data(x=x, edge_index=edge_index, y=labels_tensor)


# Custom collate function for handling variable-length sequences
def custom_collate(batch):
    input_ids = [item[0]["input_ids"].squeeze(0) for item in batch]
    attention_mask = [item[0]["attention_mask"].squeeze(0) for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)

    # Pad input_ids and attention_mask
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {"input_ids": input_ids, "attention_mask": attention_mask}, labels

def save_model(model, tokenizer, save_path):
    # Save model state_dict
    torch.save(model.state_dict(), save_path + "_model.pth")
    
    # Save tokenizer using pickle
    with open(save_path + "_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    print(f"Model and tokenizer saved to {save_path}.")

def load_model(model_class, tokenizer_class, model_init_args, save_path, device):
    # Load model state_dict
    model = model_class(**model_init_args).to(device)
    model.load_state_dict(torch.load(save_path + "_model.pth"))
    
    # Load tokenizer using pickle
    with open(save_path + "_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    print(f"Model and tokenizer loaded from {save_path}.")
    return model, tokenizer

# Main Pipeline
if __name__ == "__main__":
    # Load Labels and Groups
    labels_path = 'labels.json'  # Path to your uploaded JSON file
    labels, groups = load_labels(labels_path)
    num_classes = len(groups)
    print(f'num classes {num_classes}')

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Prepare Dataset
    dataset = AttributeDataset(labels, tokenizer, groups)
    train_samples, val_samples, train_labels, val_labels = train_test_split(
        dataset.samples, dataset.labels, test_size=0.2, random_state=42
    )

    # Graph Construction
    # max_nodes = len(train_samples)  # Limit for graph nodes
    # train_graph = construct_graph(train_samples, train_labels, tokenizer, max_nodes)
    # val_graph = construct_graph(val_samples, val_labels, tokenizer, max_nodes)

    # DataLoader
    # Create a list of Data objects (one for each graph)
    # train_graphs = [
    #     construct_graph([sample], [label], tokenizer, max_nodes=1)
    #     for sample, label in zip(train_samples, train_labels)
    # ]

    # val_graphs = [
    #     construct_graph([sample], [label], tokenizer, max_nodes=1)
    #     for sample, label in zip(val_samples, val_labels)
    # ]

    # # Use GeometricDataLoader for variable-sized graphs
    # train_loader = GeometricDataLoader(train_graphs, batch_size=16, shuffle=True)
    # # val_loader = GeometricDataLoader(val_graphs, batch_size=16)

    # # Check data loader outputs
    # for batch in train_loader:
    #     print(batch)

    # Model, Optimizer, Criterion
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertGNNClassifier("bert-base-uncased", gnn_hidden_dim=128, num_classes=num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Initialize BERT model
    bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)

    # Precompute embeddings for training and validation samples
    print("Computing BERT embeddings...")
    train_embeddings = compute_bert_embeddings(train_samples, tokenizer, bert_model, device)
    val_embeddings = compute_bert_embeddings(val_samples, tokenizer, bert_model, device)

    # Debug: Check embeddings
    print(f"Train embeddings shape: {train_embeddings.shape}")
    print(f"Validation embeddings shape: {val_embeddings.shape}")

    # Construct graphs using embeddings
    train_graph = construct_graph(train_samples, train_labels, train_embeddings, max_nodes=len(train_samples))
    val_graph = construct_graph(val_samples, val_labels, val_embeddings, max_nodes=len(val_samples))


    # Create DataLoaders
    train_loader = GeometricDataLoader([train_graph], batch_size=16, shuffle=True)
    val_loader = GeometricDataLoader([val_graph], batch_size=16)

    # Training and evaluation
    for epoch in range(5):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        val_accuracy = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    

    save_path = "bert_gnn_classifier"

    # Save the model and tokenizer
    save_model(model, tokenizer, save_path)

