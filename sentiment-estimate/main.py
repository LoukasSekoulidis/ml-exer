#%% 
from datasets import load_dataset
from transformers import AutoTokenizer, AdamW, BertForSequenceClassification, BertConfig
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
data = load_dataset('dair-ai/emotion')
config = BertConfig.from_pretrained('bert-base-uncased', num_labels=6)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
optimizer = AdamW(model.parameters(), lr=2e-5)


# %%
train_data = data['train']
validation_data = data['validation']
test_data = data['test']

# %%

def tokenization(data):
  return tokenizer(data['text'], padding='max_length', truncation=True, max_length=128, return_tensors='pt')

# %%
train_data = train_data.map(tokenization, batched=True)
validation_data = validation_data.map(tokenization, batched=True)
test_data = test_data.map(tokenization, batched=True)

# %%
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
train_data.format['type']
validation_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
validation_data.format['type']
test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_data.format['type']

batch_size = 16
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

# %%
# Training function with logging and progress tracking
def train_model(model, train_dataloader, optimizer, epochs=3):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['label']
            )
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer.step()
            
            # Calculate loss and accuracy
            epoch_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == batch['label'])
            total_predictions += batch['label'].size(0)
            
            # Update progress bar and log
            progress_bar.set_postfix(loss=epoch_loss/(total_predictions//batch_size),
                                     accuracy=correct_predictions.item()/total_predictions)
            logger.info(f"Epoch {epoch+1}, Batch {total_predictions//batch_size}, Loss: {epoch_loss/(total_predictions//batch_size):.4f}, Accuracy: {correct_predictions.item()/total_predictions:.4f}")

        avg_loss = epoch_loss / len(train_dataloader)
        avg_accuracy = correct_predictions.item() / total_predictions
        logger.info(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}")
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}")

# Call the training function
train_model(model, train_dataloader, optimizer, epochs=3)
