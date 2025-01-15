import torch
import pickle
import json
from transformers import BertTokenizer, BertModel
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader
from full_pipeline import BertGNNClassifier, construct_graph, compute_bert_embeddings  # Import from your pipeline

# Function to load the model and tokenizer
def load_model_and_tokenizer(model_path, gnn_hidden_dim, num_classes, device):
    # Load model
    model = BertGNNClassifier(gnn_hidden_dim=gnn_hidden_dim, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # Load tokenizer
    with open('bert_gnn_classifier_tokenizer.pkl', "rb") as f:
        tokenizer = pickle.load(f)
    
    print(f"Loaded model and tokenizer from {model_path}")
    return model, tokenizer

# Function to classify and preserve all attributes
def classify_and_preserve(input_json, model, tokenizer, bert_model, device, max_nodes=100):
    # Extract all attribute names and values
    all_attributes = {key: value for key, value in input_json.items()}
    
    # Get attribute names
    attribute_names = list(all_attributes.keys())
    attribute_values = list(all_attributes.values())
    
    # Compute embeddings for attribute names
    embeddings = compute_bert_embeddings(attribute_names, tokenizer, bert_model, device)
    
    # Construct a graph for classification
    graph = construct_graph(attribute_names, [0] * len(attribute_names), embeddings, max_nodes=max_nodes)
    loader = GeometricDataLoader([graph], batch_size=1)
    
    # Predict class names
    predictions = []
    with torch.no_grad():
        for batch in loader:
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            outputs = model(x, edge_index)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
    
    # Map predictions to their classes or "undefined"
    class_map = {0: "frequency", 1: "modulation", 2: "battery_level"}  # Add your real class mappings
    output_json = {}

    for attr_name, attr_value, pred_class in zip(attribute_names, attribute_values, predictions):
        class_name = class_map.get(pred_class, "undefined")
        if class_name not in output_json:
            output_json[class_name] = {}
        output_json[class_name][attr_name] = attr_value
    
    # Add unprocessed keys directly to the "undefined" class
    for key in input_json:
        if key not in output_json:
            if "undefined" not in output_json:
                output_json["undefined"] = {}
            output_json["undefined"][key] = input_json[key]
    
    return output_json

# Main script
if __name__ == "__main__":
    import sys

    # Check command-line arguments
    if len(sys.argv) < 4:
        print("Usage: python full_pipeline.py <model_path> <input_json_file> <output_json_file>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    input_json_file = sys.argv[2]
    output_json_file = sys.argv[3]

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model and tokenizer
    num_classes = 43  # Update this based on your dataset
    gnn_hidden_dim = 128
    model, tokenizer = load_model_and_tokenizer(model_path, gnn_hidden_dim, num_classes, device)
    
    # Load BERT model
    bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
    bert_model.eval()
    # model.eval()
    
    # Load input JSON
    with open(input_json_file, "r") as f:
        input_json = json.load(f)
    
    # Classify and unify JSON
    output_json = classify_and_preserve(input_json, model, tokenizer, bert_model, device)
    
    # Save output JSON
    with open(output_json_file, "w") as f:
        json.dump(output_json, f, indent=4)
    
    print(f"Unified JSON saved to {output_json_file}")
