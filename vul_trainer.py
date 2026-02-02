import torch
import numpy as np
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from vul_detector import FocalLoss

class VulTrainerManual:
    def __init__(self, model, train_loader, val_loader, device, 
                 class_weights=None, learning_rate=2e-5, num_epochs=10,
                 loss_type="cross_entropy", focal_gamma=2.0):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        
        # Setup weighted loss
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)
        else:
            self.class_weights = None

        loss_type = loss_type.lower()
        if loss_type not in {"cross_entropy", "focal"}:
            raise ValueError("loss_type must be 'cross_entropy' or 'focal'")

        if loss_type == "cross_entropy":
            weight = self.class_weights.to(device) if self.class_weights is not None else None
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            self.criterion = FocalLoss(gamma=focal_gamma, weight=self.class_weights)
            self.criterion = self.criterion.to(device)
            
        # Optimizer and scheduler
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(train_loader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # For early stopping
        self.best_f1 = 0
        self.patience = 5
        self.patience_counter = 0
        
    def compute_metrics(self, predictions, labels):
        """Calculate per-class metrics to monitor imbalance."""

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        accuracy = accuracy_score(labels, predictions)
        
        # Macro average (better for imbalanced data)
        macro_f1 = np.mean(f1)
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        
        return {
            'accuracy': accuracy,
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1,
            'class_precision': precision.tolist(),
            'class_recall': recall.tolist(),
            'class_f1': f1.tolist()
        }
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch.pop('labels')
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(**batch)
            logits = outputs.logits
            
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
            batch['labels'] = labels  # restore for safety if dataloader reuses dict
        
        # Calculate training metrics
        metrics = self.compute_metrics(np.array(all_preds), np.array(all_labels))
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss, metrics
    
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch.pop('labels')
                outputs = self.model(**batch)
                logits = outputs.logits
                loss = self.criterion(logits, labels)

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.detach().cpu().tolist())
                all_labels.extend(labels.detach().cpu().tolist())
                total_loss += loss.item()

        metrics = self.compute_metrics(np.array(all_preds), np.array(all_labels))
        avg_loss = total_loss / max(len(self.val_loader), 1)
        metrics['loss'] = avg_loss

        return metrics
    
    def train(self):
        for epoch in range(self.num_epochs):
            # Training
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate()
            
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print(f"Train Precision: {train_metrics['precision']:.4f}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train F1: {train_metrics['f1']:.4f}")
            print(f"Train Recall: {train_metrics['recall']:.4f}")
            print(f"Train F1 per class: {train_metrics['class_f1']}")
            print(f"Val Precision: {val_metrics['precision']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")
            print(f"Val Recall: {val_metrics['recall']:.4f}")
            print(f"Val F1 per class: {val_metrics['class_f1']}")
            
            # Early stopping based on F1 score
            if val_metrics['f1'] > self.best_f1:
                self.best_f1 = val_metrics['f1']
                self.patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), f"best_model_epoch_{epoch+1}.pt")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break