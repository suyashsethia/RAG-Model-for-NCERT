import torch
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# Load Dataset
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Mapping the labels to numerical values
label_dict = {'Biology': 0, 'Physics': 1, 'Chemistry': 2}
train_df['label'] = train_df['Topic'].map(label_dict)
test_df['label'] = test_df['Topic'].map(label_dict)

# Custom Dataset class for tokenizing
class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.df = df
        self.tokenizer = tokenizer
        self.texts = df['Comment'].values
        self.labels = df['label'].values
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
# Initialize tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
model.to(device)
# Create Dataset objects
train_dataset = CustomDataset(train_df, tokenizer)
test_dataset = CustomDataset(test_df, tokenizer)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=False,
)


# Define accuracy as the metric
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(f"Test Accuracy: {results['eval_accuracy']}")

# Save the fine-tuned model
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')

#Inference function
def predict(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidences = probs[0].tolist()
        predicted_class = torch.argmax(probs).item()
    return {
        'predicted_class': list(label_dict.keys())[predicted_class],
        'confidence_scores': dict(zip(list(label_dict.keys()), confidences))
    }

# Example inference
text = "Machine learning is very important."
prediction = predict(text)
print(prediction)