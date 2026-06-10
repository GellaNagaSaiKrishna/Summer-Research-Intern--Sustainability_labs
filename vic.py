import warnings
import mne
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from pathlib import Path
import re
import os
import shutil
import gc
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.metrics import balanced_accuracy_score

if __name__ == '__main__':
    # %% [code] {"execution":{"iopub.status.busy":"2026-06-04T09:37:02.785759Z","iopub.execute_input":"2026-06-04T09:37:02.785980Z","iopub.status.idle":"2026-06-04T09:37:10.921570Z","shell.execute_reply.started":"2026-06-04T09:37:02.785958Z","shell.execute_reply":"2026-06-04T09:37:10.920671Z"},"jupyter":{"outputs_hidden":false}}
    # ============================================
    # KAGGLE ABLATION STUDY (DATA EFFICIENCY)
    # ============================================
    
    warnings.filterwarnings('ignore')
    
    
    mne.set_log_level('WARNING')
    data_dir = Path('/home/gella.saikrishna/sleep-edf-database-expanded-1.0.0/sleep-edf-database-expanded-1.0.0/sleep-cassette')
    CACHE_DIR = Path('eeg_cache')
    if not CACHE_DIR.exists(): CACHE_DIR.mkdir(parents=True)
    
    #---
    
    
    # %% [code] {"execution":{"iopub.status.busy":"2026-06-04T09:37:10.923694Z","iopub.execute_input":"2026-06-04T09:37:10.924183Z","iopub.status.idle":"2026-06-04T09:37:10.949690Z","shell.execute_reply.started":"2026-06-04T09:37:10.924155Z","shell.execute_reply":"2026-06-04T09:37:10.947742Z"},"jupyter":{"outputs_hidden":false}}
    # ============================================
    # 1. MNE PROCESSING ENGINE (Saves directly to disk)
    # ============================================
    def match_files(data_dir):
        psg_files = list(data_dir.glob('*-PSG.edf'))
        hyp_files = list(data_dir.glob('*Hypnogram*.edf'))
        psg_dict = {re.search(r'SC(\d+)', f.name).group(1): f for f in psg_files if re.search(r'SC(\d+)', f.name)}
        hyp_dict = {re.search(r'SC(\d+)', f.name).group(1): f for f in hyp_files if re.search(r'SC(\d+)', f.name)}
        return [(psg_dict[sid], hyp_dict[sid]) for sid in psg_dict if sid in hyp_dict]
    
    # More comprehensive mapping
    SLEEP_STAGE_MAPPING = {
        "Sleep stage W": 0, "W": 0, "Wake": 0,
        "Sleep stage 1": 1, "1": 1, "Stage 1": 1,
        "Sleep stage 2": 2, "2": 2, "Stage 2": 2,
        "Sleep stage 3": 3, "3": 3, "Stage 3": 3,
        "Sleep stage 4": 3, "4": 3, "Stage 4": 3,
        "Sleep stage R": 4, "R": 4, "REM": 4, "Sleep stage REM": 4
    }
    
    
    
    def process_single_recording(args):
        psg_file, hyp_file = args
        target_channels = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"]
        subj_id = psg_file.name.split('-')[0][:6] 
        
        # Skip if already cached
        if (CACHE_DIR / f"{subj_id}_X.npy").exists() and (CACHE_DIR / f"{subj_id}_y.npy").exists():
            y = np.load(CACHE_DIR / f"{subj_id}_y.npy")
            return subj_id, len(y)
    
        try:
            raw_psg = mne.io.read_raw_edf(str(psg_file), preload=False)
            annotations = mne.read_annotations(str(hyp_file))
            raw_psg.set_annotations(annotations)
    
            # Match channels
            avail = []
            for t in target_channels:
                t_clean = re.sub(r"[ \-]", "", t).lower()
                for ch in raw_psg.ch_names:
                    ch_clean = re.sub(r"[ \-]", "", ch).lower()
                    if t_clean in ch_clean or ch_clean in t_clean:
                        avail.append(ch)
                        break
            if len(avail) != 4:
                return None
    
            raw_psg.pick_channels(avail)
            raw_psg.load_data() 
            raw_psg.filter(l_freq=0.5, h_freq=35.0, fir_design='firwin')
            if raw_psg.info['sfreq'] != 100:
                raw_psg.resample(100)
    
            unique_desc = set(a['description'] for a in annotations)
            event_id = {desc: SLEEP_STAGE_MAPPING[desc] for desc in unique_desc if desc in SLEEP_STAGE_MAPPING}
            events, _ = mne.events_from_annotations(raw_psg, event_id=event_id, chunk_duration=30.0)
            if len(events) == 0:
                return None
    
            epochs = mne.Epochs(raw_psg, events, tmin=0.0, tmax=30.0 - 1/raw_psg.info['sfreq'],
                                baseline=None, preload=True)
            X = epochs.get_data().astype(np.float32)  
            y = events[:, -1].astype(np.int64)
    
            # ✅ Per‑epoch, per‑channel standardization (z‑score)
            for epoch_idx in range(X.shape[0]):
                for c in range(X.shape[1]):
                    epoch_mean = X[epoch_idx, c, :].mean()
                    epoch_std = X[epoch_idx, c, :].std()
                    if epoch_std > 0:
                        X[epoch_idx, c, :] = (X[epoch_idx, c, :] - epoch_mean) / epoch_std
    
            # Cleanup **after** all processing
            del raw_psg, epochs, annotations
            gc.collect()
    
            np.save(CACHE_DIR / f"{subj_id}_X.npy", X)
            np.save(CACHE_DIR / f"{subj_id}_y.npy", y)
            return subj_id, len(y)
    
        except Exception as e:
            print(f"Error processing {subj_id}: {e}")
            return None
    
    # ============================================
    # 2. ZERO-RAM LAZY DATASET
    # ============================================
    class LazyMNESleepDataset(Dataset):
        def __init__(self, subject_ids, augment=False):
            self.augment = augment
            self.index_map = []
            self.labels = []
            
            for subj in subject_ids:
                y_data = np.load(CACHE_DIR / f"{subj}_y.npy")
                for i, label in enumerate(y_data):
                    self.index_map.append((subj, i))
                    self.labels.append(label)
                    
            self.labels = np.array(self.labels)
            self.mmap_cache = {} 
            
        def __len__(self):
            return len(self.index_map)
        
        def __getitem__(self, idx):
            subj, local_idx = self.index_map[idx]
            
            if subj not in self.mmap_cache:
                self.mmap_cache[subj] = np.load(CACHE_DIR / f"{subj}_X.npy", mmap_mode='r')
                
            epoch = torch.FloatTensor(self.mmap_cache[subj][local_idx])
            
            if self.augment:
                epoch = epoch.clone()
                if torch.rand(1).item() < 0.2: epoch += torch.randn_like(epoch) * 0.05
                if torch.rand(1).item() < 0.2: epoch *= torch.empty(epoch.shape[0], 1).uniform_(0.8, 1.2)
                    
            return epoch, self.labels[idx]
    
    def get_sampler(dataset):
        c_weights = compute_class_weight('balanced', classes=np.unique(dataset.labels), y=dataset.labels)
        s_weights = np.array([c_weights[int(l)] for l in dataset.labels])
        return WeightedRandomSampler(weights=torch.DoubleTensor(s_weights), num_samples=len(s_weights), replacement=True)
    
    #---
    
    
    # %% [code] {"execution":{"iopub.status.busy":"2026-06-04T09:37:10.950976Z","iopub.execute_input":"2026-06-04T09:37:10.951405Z","iopub.status.idle":"2026-06-04T09:37:10.987422Z","shell.execute_reply.started":"2026-06-04T09:37:10.951379Z","shell.execute_reply":"2026-06-04T09:37:10.986350Z"},"jupyter":{"outputs_hidden":false}}
    # ============================================
    # 3. OLD RESNET ARCHITECTURE
    # ============================================
    class DilatedConvBlock1D(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
            super().__init__()
            padding = dilation * (kernel_size - 1) // 2 
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=False)
            self.bn = nn.BatchNorm1d(out_channels)
            self.relu = nn.ReLU(inplace=True)
        def forward(self, x): return self.relu(self.bn(self.conv(x)))
    
    class DilatedResidualBlock1D(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1, dilation=1, downsample=None):
            super().__init__()
            self.conv1 = DilatedConvBlock1D(in_channels, out_channels, stride=stride, dilation=dilation)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
            self.bn2 = nn.BatchNorm1d(out_channels)
            self.downsample = downsample
            self.relu = nn.ReLU(inplace=True)
        def forward(self, x):
            identity = self.downsample(x) if self.downsample is not None else x
            return self.relu(self.bn2(self.conv2(self.conv1(x))) + identity)
    
    class ResNet18_1D(nn.Module):
        def __init__(self, in_channels=4, base_filters=16, hidden_dim=128, dropout_p=0.5):
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels, base_filters, kernel_size=15, stride=2, padding=7, bias=False)
            self.bn1 = nn.BatchNorm1d(base_filters)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            
            self.layer1 = self._make_layer(base_filters, base_filters, 2, stride=1, dilation=1)
            self.layer2 = self._make_layer(base_filters, base_filters*2, 2, stride=2, dilation=2)
            self.layer3 = self._make_layer(base_filters*2, base_filters*4, 2, stride=2, dilation=4)
            
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.dropout = nn.Dropout(p=dropout_p)
            self.fc = nn.Linear(base_filters*4, hidden_dim) 
        
        def _make_layer(self, in_channels, out_channels, blocks, stride, dilation):
            downsample = None
            if stride != 1 or in_channels != out_channels:
                downsample = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(out_channels)
                )
            layers = [DilatedResidualBlock1D(in_channels, out_channels, stride, dilation, downsample)]
            for _ in range(1, blocks): layers.append(DilatedResidualBlock1D(out_channels, out_channels, dilation=dilation))
            return nn.Sequential(*layers)
        
        def forward(self, x):
            x = self.layer3(self.layer2(self.layer1(self.maxpool(self.relu(self.bn1(self.conv1(x)))))))
            return self.fc(self.dropout(self.avgpool(x).view(x.size(0), -1)))
    
    class SleepStageClassifier(nn.Module):
        def __init__(self, encoder, n_classes=5, hidden_dim=128):
            super().__init__()
            self.encoder = encoder
            self.classifier = nn.Linear(hidden_dim, n_classes)
        def forward(self, x):
            features = self.encoder(x)
            return self.classifier(features), features
    
    class VICRegLoss(nn.Module):
        def __init__(self, lambda_var=25.0, lambda_cov=25.0, lambda_inv=1.0, eps=1e-4):
            super().__init__()
            self.lambda_var, self.lambda_cov, self.lambda_inv, self.eps = lambda_var, lambda_cov, lambda_inv, eps
        def forward(self, z1, z2):
            std_z1, std_z2 = torch.sqrt(z1.var(0) + self.eps), torch.sqrt(z2.var(0) + self.eps)
            var_loss = (torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))) / 2
            z1_c, z2_c = z1 - z1.mean(0), z2 - z2.mean(0)
            cov_z1, cov_z2 = (z1_c.T @ z1_c) / (z1.shape[0]-1), (z2_c.T @ z2_c) / (z2.shape[0]-1)
            cov_loss = (cov_z1.fill_diagonal_(0).pow(2).sum()/z1.shape[1] + cov_z2.fill_diagonal_(0).pow(2).sum()/z2.shape[1]) / 2
            inv_loss = F.mse_loss(z1, z2)
            return self.lambda_var*var_loss + self.lambda_cov*cov_loss + self.lambda_inv*inv_loss, var_loss, cov_loss, inv_loss
    
    # ============================================
    # 4. ABLATION TRAINING UTILITIES
    # ============================================
    
    def strong_eeg_augment(x):
        """Applies asymmetric transformations to force the SSL encoder to learn real features."""
        x_aug = x.clone()
        b, c, t = x.shape
        
        # 1. Random Amplitude Scaling (0.5x to 2.0x per channel)
        scales = torch.empty(b, c, 1, device=x.device).uniform_(0.5, 2.0)
        x_aug = x_aug * scales
        
        # 2. Additive Gaussian Noise
        x_aug = x_aug + (torch.randn_like(x_aug) * 0.1)
        
        # 3. Channel Masking (Randomly zero out 1 channel in 30% of batches to force cross-channel dependency)
        if torch.rand(1).item() < 0.3:
            ch_idx = torch.randint(0, c, (1,)).item()
            x_aug[:, ch_idx, :] = 0.0
            
        return x_aug
    
    def pretrain_vicreg(loader_A, device):
        print("\n" + "="*50 + "\nPRETRAINING ENCODER ON 70 SUBJECTS\n" + "="*50)
        encoder = ResNet18_1D(in_channels=4, hidden_dim=128).to(device)
        optimizer = optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40) 
        vicreg = VICRegLoss()
        
        for epoch in range(40):
            total_loss = 0
            encoder.train()
            for data, _ in loader_A:
                data = data.to(device)
                optimizer.zero_grad()
                
                # Use strong, randomized augmentations for each view
                z1 = encoder(strong_eeg_augment(data))
                z2 = encoder(strong_eeg_augment(data))
                
                loss, _, _, _ = vicreg(z1, z2)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            scheduler.step()
            if (epoch + 1) % 10 == 0: 
                print(f"Pretrain Epoch {epoch+1:02d} | VICReg Loss: {total_loss / len(loader_A):.4f}")
                
        return encoder
    
    #---
    
    
    # %% [code] {"execution":{"iopub.status.busy":"2026-06-04T09:37:10.988599Z","iopub.execute_input":"2026-06-04T09:37:10.988916Z","iopub.status.idle":"2026-06-04T09:37:11.018764Z","shell.execute_reply.started":"2026-06-04T09:37:10.988876Z","shell.execute_reply":"2026-06-04T09:37:11.017702Z"},"jupyter":{"outputs_hidden":false}}
    # =====================================================================
    # MULTI-METRIC TRACKING ABLATION PIPELINE
    # =====================================================================
    
    
    def evaluate_all_metrics(model, loader, criterion, device):
        """Computes Loss, Balanced Accuracy, and Weighted F1 score."""
        model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for data, labels in loader:
                data, labels = data.to(device), labels.to(device)
                logits, _ = model(data)
                loss = criterion(logits, labels)
                total_loss += loss.item() * data.size(0)
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        total_samples = len(loader.dataset)
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        bal_acc = balanced_accuracy_score(all_labels, all_preds) if all_labels else 0
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) if all_labels else 0
        
        return avg_loss, bal_acc, f1
    
    def train_epoch(model, loader, optimizer, criterion, device):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(data)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        avg_loss = total_loss / len(loader)
        bal_acc = balanced_accuracy_score(all_labels, all_preds) if all_labels else 0
        return avg_loss, bal_acc
    
    def run_supervised_baseline(train_loader, val_loader, test_loader, epochs, class_weights, device):
        scratch_encoder = ResNet18_1D(in_channels=4, hidden_dim=128).to(device)
        model = SleepStageClassifier(scratch_encoder, n_classes=5, hidden_dim=128).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
        
        best_val_f1 = -1.0
        best_model_state = None
        
        for epoch in range(epochs):
            train_epoch(model, train_loader, optimizer, criterion, device)
            _, _, val_f1 = evaluate_all_metrics(model, val_loader, criterion, device)
            scheduler.step()
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                
        model.load_state_dict(best_model_state)
        
        tr_loss, tr_bal_acc, tr_f1 = evaluate_all_metrics(model, train_loader, criterion, device)
        va_loss, va_bal_acc, va_f1 = evaluate_all_metrics(model, val_loader, criterion, device)
        te_loss, te_bal_acc, te_f1 = evaluate_all_metrics(model, test_loader, criterion, device)
        
        return {
            'train_loss': tr_loss, 'train_acc': tr_bal_acc, 'train_f1': tr_f1,
            'val_loss': va_loss,   'val_acc': va_bal_acc,   'val_f1': va_f1,
            'test_loss': te_loss,  'test_acc': te_bal_acc,  'test_f1': te_f1
        }
    
    def run_contrastive_downstream(model_type, base_encoder, train_loader, val_loader, test_loader, epochs, class_weights, device):
        """Handles Freeze/Finetune downstream tasks and returns full train/val/test metrics."""
        assert model_type in ['freeze', 'finetune']
        
        downstream_encoder = ResNet18_1D(in_channels=4, hidden_dim=128).to(device)
        downstream_encoder.load_state_dict(base_encoder.state_dict())
        
        model = SleepStageClassifier(downstream_encoder, n_classes=5, hidden_dim=128).to(device)
        
        if model_type == 'freeze':
            for param in model.encoder.parameters(): 
                param.requires_grad = False
            optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
        else:  
            optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
        
        best_val_f1 = -1.0
        best_model_state = None
        
        for epoch in range(epochs):
            train_epoch(model, train_loader, optimizer, criterion, device)
            _, _, val_f1 = evaluate_all_metrics(model, val_loader, criterion, device)
            scheduler.step()
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                
        model.load_state_dict(best_model_state)
        
        tr_loss, tr_acc, tr_f1 = evaluate_all_metrics(model, train_loader, criterion, device)
        va_loss, va_acc, va_f1 = evaluate_all_metrics(model, val_loader, criterion, device)
        te_loss, te_acc, te_f1 = evaluate_all_metrics(model, test_loader, criterion, device)
        
        return {
            'train_loss': tr_loss, 'train_acc': tr_acc, 'train_f1': tr_f1,
            'val_loss': va_loss,   'val_acc': va_acc,   'val_f1': va_f1,
            'test_loss': te_loss,  'test_acc': te_acc,  'test_f1': te_f1
        }
    
    #---
    
    
    # %% [code] {"execution":{"iopub.status.busy":"2026-06-04T09:37:11.020024Z","iopub.execute_input":"2026-06-04T09:37:11.021067Z","iopub.status.idle":"2026-06-04T09:37:11.042933Z","shell.execute_reply.started":"2026-06-04T09:37:11.021022Z","shell.execute_reply":"2026-06-04T09:37:11.041432Z"},"jupyter":{"outputs_hidden":false}}
    # import shutil
    # shutil.rmtree("/kaggle/working/eeg_cache")
    
    # %% [code] {"execution":{"iopub.status.busy":"2026-06-04T09:37:11.044280Z","iopub.execute_input":"2026-06-04T09:37:11.044630Z","iopub.status.idle":"2026-06-04T09:45:28.128518Z","shell.execute_reply.started":"2026-06-04T09:37:11.044582Z","shell.execute_reply":"2026-06-04T09:45:28.127516Z"},"jupyter":{"outputs_hidden":false}}
    # =====================================================================
    # MAIN ABLATION EXECUTION LOOP (COMPLETE & UNBROKEN)
    # =====================================================================
    
    if __name__ == "__main__":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}\n")
        batch_size = 128
    
        # 1. Parse EDFs using MNE (Exactly 90 pairs)
        pairs = match_files(data_dir)[:90] 
        print(f"Extracting {len(pairs)} recordings to disk buffer (using 4 Cores)...")
        
        metadata = []
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_single_recording, p) for p in pairs]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
                res = future.result()
                if res is not None: metadata.append(res)
    
        # 2. Extract unique subjects and shuffle
        unique_subjects = list(set([m[0] for m in metadata]))
        np.random.seed(42)
        np.random.shuffle(unique_subjects)
    
        # 3. Global Base Splits (A=70, C=10, D=10)
        pretrain_subjs = unique_subjects[0:70]
        val_subjs = unique_subjects[70:80]    # Dataset C
        test_subjs = unique_subjects[80:90]   # Dataset D
    
        print("\nGlobal Splits Generated:")
        print(f"  Dataset A (Pretraining Pool): {len(pretrain_subjs)} subjects")
        print(f"  Dataset C (Validation):       {len(val_subjs)} subjects")
        print(f"  Dataset D (Test):             {len(test_subjs)} subjects")
    
    #---
    
    
    # %% [code] {"execution":{"iopub.status.busy":"2026-06-04T09:45:28.130871Z","iopub.execute_input":"2026-06-04T09:45:28.131443Z","iopub.status.idle":"2026-06-04T10:35:37.076700Z","shell.execute_reply.started":"2026-06-04T09:45:28.131412Z","shell.execute_reply":"2026-06-04T10:35:37.075702Z"},"jupyter":{"outputs_hidden":false}}
    # 4. Build Validation (C) and Test (D) Loaders
    # No augmentation for validation or testing!
    dataset_C = LazyMNESleepDataset(val_subjs, augment=False)
    dataset_D = LazyMNESleepDataset(test_subjs, augment=False)
    
    loader_C = DataLoader(dataset_C, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    loader_D = DataLoader(dataset_D, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # 5. PRETRAIN THE VICREG ENCODER
    # Use all 70 subjects in Pool A for the unlabeled contrastive task
    dataset_A = LazyMNESleepDataset(pretrain_subjs, augment=True)
    loader_A = DataLoader(dataset_A, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    print("\nStarting VICReg Pretraining on 70 subjects...")
    pretrained_encoder = pretrain_vicreg(loader_A, device)
    
    #---
    
    
    # 6. SETUP ABLATION TRACKING
    subset_sizes = [5, 10, 20, 50]
    epoch_scaler = {5: 50, 10: 40, 20: 30, 50: 20} 
    
    ablation_results = {
        size: {'supervised': {}, 'freeze': {}, 'finetune': {}} for size in subset_sizes
    }
    
    print("\n" + "="*70 + "\nSTARTING COMPREHENSIVE MULTI-METRIC ABLATION STUDY\n" + "="*70)
    
    #---
    
    
    for n in subset_sizes:
        print(f"\n>>> Labeled Subset Size: {n} Subjects <<<")
        
        # Sample N subjects from the pretraining pool to act as our Labeled Subset B
        subset_B_subjs = pretrain_subjs[:n] 
        dataset_B = LazyMNESleepDataset(subset_B_subjs, augment=True)
        loader_B = DataLoader(dataset_B, batch_size=batch_size, sampler=get_sampler(dataset_B), num_workers=2, pin_memory=True)
        
        c_weights = compute_class_weight('balanced', classes=np.unique(dataset_B.labels), y=dataset_B.labels)
        epochs = epoch_scaler[n]
    
        # --- Run 1: Pure Supervised Baseline (Scratch) ---
        print(f"  [1/3] Training Supervised Baseline from scratch ({epochs} epochs)...")
        ablation_results[n]['supervised'] = run_supervised_baseline(loader_B, loader_C, loader_D, epochs, c_weights, device)
        
        # --- Run 2: Linear Probe (Freeze) ---
        print(f"  [2/3] Training Frozen Linear Probe ({epochs} epochs)...")
        ablation_results[n]['freeze'] = run_contrastive_downstream('freeze', pretrained_encoder, loader_B, loader_C, loader_D, epochs, c_weights, device)
        
        # --- Run 3: End-to-End Finetune ---
        print(f"  [3/3] Training End-to-End Finetune ({epochs} epochs)...")
        ablation_results[n]['finetune'] = run_contrastive_downstream('finetune', pretrained_encoder, loader_B, loader_C, loader_D, epochs, c_weights, device)
    
        # Immediate print out of test metrics for this loop step
        print(f"  Size {n} Test Metrics Summary:")
        for model_key in ['supervised', 'freeze', 'finetune']:
            res = ablation_results[n][model_key]
            print(f"{model_key.capitalize()} -> Test Bal Acc: {res['test_acc']:.4f} | Test F1: {res['test_f1']:.4f} | Test Loss: {res['test_loss']:.4f}")
    
    #---
    
    
    # 7. FINAL PLOTTING
    print("\nGenerating performance summary curves...")
    
    models = ['supervised', 'freeze', 'finetune']
    labels = ['Supervised (Scratch)', 'Contrastive + Freeze', 'Contrastive + Finetune']
    markers = ['o', 's', '^']
    
    fig, (ax1, _) = plt.subplots(1, 2, figsize=(16, 6))
    
    for model_key, label, marker in zip(models, labels, markers):
        test_accs = [ablation_results[size][model_key]['test_acc'] for size in subset_sizes]
        
        ax1.plot(subset_sizes, test_accs, marker=marker, linewidth=2, label=label)
        
    
    ax1.set_title('Data Efficiency: Test Balanced Accuracy', fontsize=13)
    ax1.set_xlabel('Number of Labeled Subjects', fontsize=11)
    ax1.set_ylabel('Balanced Accuracy', fontsize=11)
    ax1.set_xticks(subset_sizes)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()
    
    plt.tight_layout()
    plt.show()
    
    # %% [code] {"jupyter":{"outputs_hidden":false}}


##########################################################################
# This file was converted using nb2py: https://github.com/BardiaKh/nb2py #
##########################################################################