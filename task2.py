import torch
from torch.utils.data import DataLoader , Dataset
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from transformers import DataCollatorWithPadding, AdamW
from tqdm import tqdm
from tqdm import tqdm 
import os



# load task 1 model and base model
MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


""" here we can load tokenizer from task1 but we prefer to use the tokenizer from the pretrained model.
 TOKENIZER_PATH = "best_model_tokenizer"
 tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
"""
# load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
print("Tokenizer loaded successfully.")

# load the pre-trained model
print("Loading base model...")
base_model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
print("Base model loaded successfully.")



# Define the regression model class to add regressionhead to base model and get aligned with task1 model
class MoLFormerWithRegressionHead(torch.nn.Module):
    def __init__(self, base_model):
        super(MoLFormerWithRegressionHead, self).__init__()
        self.base_model = base_model
        # hidden_size = self.base_model.config.hidden_size if hasattr(self.base_model.config, "hidden_size") else 1024
        hidden_size = getattr(base_model.config, "hidden_size", 1024)
        self.regressor = torch.nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        last_hidden_state = outputs[0]  # (batch_size, seq_len, hidden_size)
        pooled = last_hidden_state[:, 0, :]  # CLS token representation
        out = self.regressor(pooled)  # (batch_size, 1)
        return out



# Initialize model and load weights for both base line and fine tuned training.
model = MoLFormerWithRegressionHead(base_model)
fine_model = MoLFormerWithRegressionHead(base_model)


"""here we are loading model from task 1, we can comment this line to use just the pretrained model with regression head too.
# Load the fine-tuned model state
MODEL_PATH = "best_supervised_model.pt"
print(f"Loading fine-tuned model from {MODEL_PATH}...")
model_state = torch.load(MODEL_PATH, map_location=device)
# model.load_state_dict(model_state, strict = False)
# print("Fine-tuned model loaded successfully.")
"""


# Move model to GPU if available
model.to(device)
model.train()
print(f"Model is now on {device} and set to train mode.")


###############################################################################################
#loading test set
###############################################################################################
# loading dataset
DATASET_PATH = "scikit-fingerprints/MoleculeNet_Lipophilicity"

print("Loading Lipophilicity test dataset...")
try:
    full_dataset = load_dataset(DATASET_PATH, split="train")
    print(f"Lipophilicity dataset loaded successfully with {len(full_dataset)} samples.")
except Exception as e:
    print(f"Error loading Lipophilicity dataset: {e}")
    exit(1)

# Convert to DataFrame for easier handling
data_df = pd.DataFrame(full_dataset)
print("Dataset converted to DataFrame.")

# Manually split into 80% train, 20% test
train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=42)

################################################################################################
#loading external dataset
#################################################################################################
# Define the path to the external dataset (consistent with final.py)
EXTERNAL_DATA_PATH = "External-Dataset_for_Task2.csv"

print("Loading external dataset...")
try:
    external_df = pd.read_csv(EXTERNAL_DATA_PATH)
    print(f"External dataset loaded successfully with {len(external_df)} samples.")
except Exception as e:
    print(f"Error loading external dataset: {e}")
    exit(1)


# Standardize external dataset column names
external_df.rename(columns={"SMILES": "SMILES", "Label": "label"}, inplace=True)

# Check if 'SMILES' and 'label' columns exist
required_columns = {"SMILES", "label"}
if not required_columns.issubset(external_df.columns):
    print(f"Error: External dataset is missing required columns: {required_columns - set(external_df.columns)}")
    exit(1)
# till here we added pretrained model and fine tuned model and dataset and train and test set and also external dataset.
#---------------------------------------------------------------------------------------



batch_size = 32  # Adjust batch size to fit memory
loss_fn = torch.nn.MSELoss() 

#create a dataset class for efficient loading
class SMILESDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        smiles = row["SMILES"]
        label = torch.tensor(row["label"], dtype=torch.float)

        encoding = self.tokenizer(
            smiles, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding["label"] = label

        return encoding

######################################################################################
#training:
######################################################################################
train_dataset = SMILESDataset(train_df, tokenizer)
train_loader = DataLoader(train_dataset, batch_size,shuffle =True)

test_dataset = SMILESDataset(test_df, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def fine_tune_and_evaluate(model, train_loader, test_loader, epochs=10, lr=2e-5, save_model_path="fine_tuned_model.pt", patience=2):
    """
    Fine-tune the pre-trained model and evaluate it with early stopping.

    Parameters:
        - model: The pre-trained model to fine-tune.
        - train_loader: DataLoader for the training dataset.
        - test_loader: DataLoader for the test dataset.
        - epochs: Max number of training epochs (default: 10).
        - lr: Learning rate for fine-tuning (default: 2e-5).
        - save_model_path: Path to save the fine-tuned model.
        - patience: Early stopping patience (default: 2 epochs).

    Returns:
        - A dictionary containing MSE, RMSE, MAE, and R² scores.
    """

    # Define loss function and optimizer
    loss_fn = torch.nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Early Stopping Setup
    best_test_loss = float("inf")  # Initialize with a high value
    best_model_state = None  # To store the best model weights
    epochs_no_improve = 0  # Track epochs without improvement

    # Fine-tuning loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", dynamic_ncols=True)

        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.squeeze(), labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix(loss=train_loss / (progress_bar.n + 1))

        avg_train_loss = train_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} completed. Avg Train Loss: {avg_train_loss:.6f}")

        # --- Evaluate on Test Set ---
        model.eval()
        test_loss = 0
        all_labels, all_preds = [], []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating Epoch {epoch+1}", dynamic_ncols=True):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.squeeze(), labels)
                test_loss += loss.item()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(outputs.squeeze().cpu().numpy())

        avg_test_loss = test_loss / len(test_loader)
        print(f"🔹 Test Loss After Epoch {epoch+1}: {avg_test_loss:.6f}")

        # Early stopping logic
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_model_state = model.state_dict()  # Save best model state
            epochs_no_improve = 0
            print(f"✅ New Best Model Found! Saving Model at Epoch {epoch+1} with Test Loss: {avg_test_loss:.6f}")
        else:
            epochs_no_improve += 1
            print(f"⚠️ No Improvement for {epochs_no_improve}/{patience} epochs.")

        if epochs_no_improve >= patience:
            print(f"🛑 Early stopping triggered at Epoch {epoch+1}. Best Test Loss: {best_test_loss:.6f}")
            break

    # Save the best model
    if best_model_state is not None:
        torch.save(best_model_state, save_model_path)
        print(f"✅ Fine-tuned model saved as '{save_model_path}' with Best Test Loss: {best_test_loss:.6f}")

    # --- Final Model Evaluation ---
    print(f"\n🔹 Evaluating Final Model on Test Set: {save_model_path}")

    model.load_state_dict(best_model_state)  # Load the best saved model
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Final Evaluation", dynamic_ncols=True):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask).squeeze().cpu().numpy()

            all_labels.extend(labels)
            all_preds.extend(outputs)

    # Compute final evaluation metrics
    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    # Print final results
    print("\n🔹 Final Model Evaluation Results:")
    print(f"✅ MSE  (Mean Squared Error): {mse:.6f}")
    print(f"✅ RMSE (Root Mean Squared Error): {rmse:.6f}")
    print(f"✅ MAE  (Mean Absolute Error): {mae:.6f}")
    print(f"✅ R² Score: {r2:.6f}")

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

# Fine-tune and evaluate with early stopping
baseline_metrics = fine_tune_and_evaluate(
    model, train_loader, test_loader, epochs=10, save_model_path="baseline_model.pt", patience=2
)


###############################################################################################
#calculating hessian inverse based on train dataset
###############################################################################################

train_dataset = SMILESDataset(train_df, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

def compute_hessian_inverse_lissa(model, train_loader, loss_fn, damping=0.1, num_samples=500):
    
    # Automatically use CUDA if available, otherwise use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔹 Using device: {device}")

    print("🔹 Starting Hessian inverse computation using LiSSA...")

    # Ensure the model is on the correct device
    model.to(device)

    # Initialize a random vector for Hessian-vector product approximation
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    random_vector = torch.randn(num_params, device=device)
    hvp = torch.zeros_like(random_vector)

    print(f"🔹 Running {num_samples} LiSSA iterations.")
    
    lambda_reg = 1e-2  # Small positive value for regularization of hessian inverse
    num_iterations = 0
    for batch in train_loader:
        if num_iterations >= num_samples:
            break
        model.zero_grad()

        # Prepare data by ensuring they are on the correct device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # Forward pass without mixed precision
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.squeeze(), labels)

        # Compute Hessian-vector product
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        grads = torch.cat([g.view(-1) for g in grads if g is not None])
        
        hvp_current = torch.autograd.grad(grads @ random_vector, model.parameters(), retain_graph=True)
        hvp_current = torch.cat([h.view(-1) for h in hvp_current if h is not None],dim=0)
        


        # LiSSA approximation update rule
        hvp += (random_vector - hvp_current - damping * hvp) / (1 + damping + lambda_reg)
        num_iterations += 1



    print("✅ Finished Hessian inverse computation.")
    print(f"🔹 Final Hessian inverse norm: {torch.norm(hvp).item():.6f}")
    print("Min HVP:", hvp.min().item(), "Max HVP:", hvp.max().item())    

    return hvp

# calling the function
hessian_inverse_vector = compute_hessian_inverse_lissa(model, train_loader, loss_fn)

# Save results
torch.save(hessian_inverse_vector, "hessian_inverse_fast.pt")

###############################################################################################
#computing ihvp that is test gradient multiplied by hessian inverse
###############################################################################################

def compute_ihvp(model, test_loader, hvp, loss_fn, device):
    """
    Parameters:
        model: The trained model.
        test_loader: DataLoader for the test set.
        hvp: The Hessian inverse-vector product computed from LiSSA.
        loss_fn: The loss function (e.g., MSELoss).
        device: The computing device (CPU/GPU).

    Returns:
        ihvp: The inverse Hessian-vector product (H⁻¹ * ∇L_test).
    """
    model.eval()  # Set model to evaluation mode

    total_test_grads = torch.zeros(hvp.shape[0]).to(device)  # Store total gradient

    # Compute test gradients over all batches
    for batch in test_loader:
        model.zero_grad()

        # Move batch data to device (GPU/CPU)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.squeeze(), labels)

        # Compute per-batch gradients
        test_grads = torch.autograd.grad(loss, model.parameters(), retain_graph=False)
        
        # Flatten gradients and sum them
        test_grads = torch.cat([g.view(-1) for g in test_grads if g is not None], dim=0)
        total_test_grads += test_grads  # Sum gradients over all batches

    # using element wise multiplication instead of matmul to avoid expensive calculation.
    ihvp = hvp * total_test_grads
    print("Sample Training Gradient:", total_test_grads[:5])
    
    ihvp = ihvp.view(-1)    # Ensure IHVP remains a 1D tensor
    ihvp = ihvp / torch.norm(ihvp)  # Normalize IHVP before use

    print("Final IHVP Shape:", ihvp.shape)

    return ihvp

test_dataset = SMILESDataset(test_df, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
loss_fn = torch.nn.MSELoss() 
#calling the function
ihvp = compute_ihvp(model, test_loader, hessian_inverse_vector,loss_fn,device)

print("IHVP Min:", ihvp.min().item(), "IHVP Max:", ihvp.max().item())
print("IHVP Norm:", torch.norm(ihvp).item())  # Overall magnitude


#############################################################################################
#now we compute influence score for external dataset
#############################################################################################
def compute_external_influence_scores(model, dataloader, loss_fn, ihvp, save_path="external_influence_scores.pt"):
    """
    Parameters:
      - model: the neural network model.
      - dataloader: DataLoader for the external dataset.
      - loss_fn: loss function.
      - ihvp: Precomputed inverse Hessian-vector product .
      - save_path: path to the file in which influence scores will be saved.
    
    Returns:
      - save_path: The path where influence scores are stored.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔹 Using device: {device}")
    
    
    
    model.to(device)
    # Get all trainable parameters (ensuring gradients are computed for the entire model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in trainable_params)
    print(f"🔹 Total trainable parameters: {num_params}")
    
    total_samples_processed = 0
    influence_scores = []  # This list will store one scalar per sample

    progress_bar = tqdm(dataloader, desc="Processing batches", dynamic_ncols=True)
    
    for batch in progress_bar:
        model.zero_grad()
        
        # Move batch to GPU
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        batch_size = input_ids.shape[0]
        
        # Process each sample in the current batch
        for i in range(batch_size):
            model.zero_grad()
            
            # Prepare a single-sample batch (add batch dimension)
            sample_input_ids = input_ids[i].unsqueeze(0)
            sample_attention_mask = attention_mask[i].unsqueeze(0)
            sample_label = labels[i].unsqueeze(0)
            
            # Forward pass for the single sample
            outputs = model(input_ids=sample_input_ids, attention_mask=sample_attention_mask)
            loss = loss_fn(outputs.squeeze(), sample_label)
            
            # Compute per-sample gradients for all trainable parameters
            grads = torch.autograd.grad(loss, trainable_params, retain_graph=False, create_graph=False)
            
            # Flatten the gradients into a single 1D vector
            sample_grads = torch.cat([g.view(-1) for g in grads if g is not None], dim=0).float()

            # Compute influence score: negative dot product between IHVP and sample gradient (Make sure both tensors are on the same device.)
            influence_score = - torch.dot(ihvp.to(device), sample_grads.to(device)) 
            influence_scores.append(influence_score.item())
            
            total_samples_processed += 1
        
        progress_bar.set_postfix(samples_processed=total_samples_processed)
    #watching some external gradients and their amount.    
    print("Sample External Gradient:", sample_grads[:5])
    print("Sample External Gradient Norm:", torch.norm(sample_grads).item())    

    # Optionally, save influence scores incrementally (here we save after each batch)
    torch.save(influence_scores, save_path)
    
    print(f"\n✅ Influence score computation completed! Saved {total_samples_processed} scores to {save_path}")
    
    return save_path



#calculating influence scores:
external_dataset = SMILESDataset(external_df, tokenizer)
external_loader = DataLoader(external_dataset, batch_size=batch_size, shuffle=False)
loss_fn = torch.nn.MSELoss() 
# Compute and save the influence scores for the external dataset samples
external_influence_scores_file = compute_external_influence_scores(
    model, external_loader, loss_fn, ihvp, save_path="external_influence_scores.pt"
)



#############################################################################################
# sort influence scores and update the train dataset
#############################################################################################
# To load the influence scores :
influence_scores = torch.load(external_influence_scores_file)
print(f"Loaded {len(influence_scores)} influence scores.")

influence_scores_np = np.array(influence_scores)
#some prints to analyze influence scores
print("\n🔹 Influence Score Statistics:")
print(f"Min Influence Score: {influence_scores_np.min()}")
print(f"Max Influence Score: {influence_scores_np.max()}")
print(f"Mean Influence Score: {influence_scores_np.mean()}")
print(f"Std Influence Score: {influence_scores_np.std()}")

# Sort influence scores 
sorted_indices = np.argsort(influence_scores_np)  # Sorted in ascending order (most negative first)

print("\n🔹 Top 10 Most Negative Influence Scores (Expected Most Helpful)")
for idx in sorted_indices[:10]:  # 10 most negative
    print(f"Sample {idx} → Influence Score: {influence_scores_np[idx]:.6f}")
print("\n🔹 Top 10 Most Positive Influence Scores (Unexpectedly Harmful?)")
for idx in sorted_indices[-10:]:  # 10 most positive
    print(f"Sample {idx} → Influence Score: {influence_scores_np[idx]:.6f}")

""" here we can choose between some options for example changing number of most influentianl external samples 
or most negative and most positive ones for checking the evaluations of fine tuned model.  """
# Select **top N most influential external samples**
top_neg = 100  
top_pos = 20

#  most negative 
high_impact_samples = external_df.iloc[sorted_indices[:top_neg]]  # Select most negative influence score samples

# most positive 
# high_impact_samples = external_df.iloc[sorted_indices[-top_pos:]]  # Select most POSITIVE influence scores

#most negative plus some most positive
# selected_indices = np.concatenate((sorted_indices[:top_neg], sorted_indices[-top_pos:]))
# high_impact_samples = external_df.iloc[selected_indices]

# Combine high-impact external samples with the full Lipophilicity dataset
updated_lipophilicity_df = pd.concat([train_df, high_impact_samples], ignore_index=True)

# Save the updated dataset for further analysis and fine-tuning
updated_lipophilicity_df.to_csv("updated_lipophilicity_dataset.csv", index=False)

print(f"✅ Updated dataset saved as 'updated_lipophilicity_dataset.csv' with {len(updated_lipophilicity_df)} total samples.")
print(f"🔹 Added {len(high_impact_samples)} high-impact external samples (influence < 0).")
print(f"number of full dataset is {len(updated_lipophilicity_df)}")



######################################################################################
#fine tuning and training:
######################################################################################

# combination of train df and external impactfulls
updated_dataset = SMILESDataset(updated_lipophilicity_df, tokenizer)
updated_loader = DataLoader(updated_dataset, batch_size=batch_size, shuffle=False)

#just impactful datapoints.
impactfulexternal_dataset = SMILESDataset(high_impact_samples, tokenizer)
impactfulexternal_loader = DataLoader(impactfulexternal_dataset, batch_size=batch_size, shuffle=False)

#test set for evaluation.
test_dataset = SMILESDataset(test_df, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#Fine-tune and evaluate with the **influence-score-selected training dataset on the same model**
influence_metrics = fine_tune_and_evaluate(
    model, updated_loader, test_loader, epochs=10, save_model_path="fine_tuned_with_samemodel.pt", patience=2
)

influence_metrics_and_train = fine_tune_and_evaluate(
    fine_model, updated_loader, test_loader, epochs=10, save_model_path="fine_tuned_with_newmodel.pt", patience=2
)

#Compare results
print("\n🔹 Comparison of Model Performance:")
print("Baseline Model:", baseline_metrics)
print("train+external on same model Fine-Tuned :", influence_metrics)
print("train+external on fresh model Fine-Tuned :",influence_metrics_and_train )


