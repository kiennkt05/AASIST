import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

import math

EPS = 1e-6

class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()
        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = grad_output * input.exp().detach() / (running_mean + EPS) / input.shape[0]
        return grad, None

def ema(mu, alpha, past_ema):
    return alpha * mu + (1.0 - alpha) * past_ema

def ema_loss(x, running_mean, alpha):
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    t_log = EMALoss.apply(x, running_mean)
    return t_log, running_mean

class Mine(nn.Module):
    def __init__(self, T, loss='mine', alpha=0.01, method=None):
        super().__init__()
        self.running_mean = 0
        self.loss = loss
        self.alpha = alpha
        self.method = method
        self.T = T

    def forward(self, x, z, z_marg=None):
        if z_marg is None:
            z_marg = z[torch.randperm(x.shape[0])]

        t = self.T(x, z).mean()
        t_marg = self.T(x, z_marg)

        if self.loss in ['mine']:
            second_term, self.running_mean = ema_loss(t_marg, self.running_mean, self.alpha)
        elif self.loss in ['fdiv']:
            second_term = torch.exp(t_marg - 1).mean()
        elif self.loss in ['mine_biased']:
            second_term = torch.logsumexp(t_marg, 0) - math.log(t_marg.shape[0])

        return -t + second_term

class ConcatLayer(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x, y):
        return torch.cat((x, y), self.dim)

class CustomSequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            if isinstance(input, tuple):
                input = module(*input)
            else:
                input = module(input)
        return input


from data_utils import Dataset_ASVspoof2019_train, Dataset_ASVspoof2019_devNeval, genSpoof_list
from models.AASIST import DynamicFrontend
from main import torch_load_compat, seed_worker, set_seed

class MINEAnalysis:
    def __init__(self, config_path, weights_path, device, batch_size=64):
        self.device = device
        self.batch_size = batch_size
        
        with open(config_path, "r") as f:
            self.config = json.loads(f.read())
            
        model_config = self.config["model_config"]
        # Extract frontend params
        filts = model_config["filts"]
        self.num_filters = filts[0] if isinstance(filts[0], int) else 16
        
        if "32" in str(weights_path) or self.num_filters == 16: 
            pass
        
        # We will initialize a frontend and load weights
        self.frontend = DynamicFrontend(
            filts=self.num_filters,
            first_conv=model_config["first_conv"],
            use_gabor=True,
            use_spcen=True,
            use_sm=False # We extract both real and imag before SM
        ).to(device)
        
        # Load weights
        if weights_path and os.path.exists(weights_path):
            print(f"Loading weights from {weights_path}")
            checkpoint = torch_load_compat(weights_path, map_location=device)
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
                
            # Filter only frontend weights
            frontend_state = {k.replace("frontend.", ""): v for k, v in state_dict.items() if k.startswith("frontend.")}
            
            # Check if checkpoint has different number of filters
            if "filterbank.center_freqs" in frontend_state:
                ckpt_filters = frontend_state["filterbank.center_freqs"].shape[0]
                if ckpt_filters != self.num_filters:
                    print(f"Overriding config filters ({self.num_filters}) with checkpoint filters ({ckpt_filters})")
                    self.num_filters = ckpt_filters
                    self.frontend = DynamicFrontend(
                        filts=self.num_filters,
                        first_conv=model_config["first_conv"],
                        use_gabor=True,
                        use_spcen=True,
                        use_sm=False
                    ).to(device)
                    
            self.frontend.load_state_dict(frontend_state, strict=False)
        else:
            print(f"Warning: Weights path {weights_path} does not exist. Using random weights.")
            
        self.frontend.eval()
        for param in self.frontend.parameters():
            param.requires_grad = False
            
        # Define MINE networks
        # Y is one-hot encoded -> shape (B, 2)
        y_dim = 2
        
        # M is magnitude (B, F), Z is complex (B, 2F)
        m_dim = self.num_filters
        z_dim = self.num_filters * 2
        
        # MLP for M
        self.mine_M = Mine(
            T=CustomSequential(
                ConcatLayer(),
                nn.Linear(m_dim + y_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ),
            loss='mine',
            alpha=0.01,
            method='concat'
        ).to(device)
        
        # MLP for Z
        self.mine_Z = Mine(
            T=CustomSequential(
                ConcatLayer(),
                nn.Linear(z_dim + y_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ),
            loss='mine',
            alpha=0.01,
            method='concat'
        ).to(device)
        
        self.opt_M = torch.optim.Adam(self.mine_M.parameters(), lr=1e-4)
        self.opt_Z = torch.optim.Adam(self.mine_Z.parameters(), lr=1e-4)

    def extract_features(self, x):
        """
        Extract M and Z using Global Max Pooling 1D.
        """
        x = x.unsqueeze(1) # (B, 1, T)
        
        with torch.no_grad():
            out_complex = self.frontend.filterbank(x)
            
            # Split real and imaginary parts
            out_real, out_imag = torch.chunk(out_complex, 2, dim=1)
            
            # Z: complex representation
            # M: magnitude representation R^2 + I^2
            M = torch.square(out_real) + torch.square(out_imag)
            Z = out_complex
            
            # Global Max Pooling over time dimension
            M_pooled = F.adaptive_max_pool1d(M, 1).squeeze(2) # (B, F)
            Z_pooled = F.adaptive_max_pool1d(Z, 1).squeeze(2) # (B, 2F)
            
        return M_pooled, Z_pooled

    def train(self, trn_loader, epochs=50):
        print("Starting MINE training...")
        for epoch in range(epochs):
            loss_M_total = 0.0
            loss_Z_total = 0.0
            steps = 0
            
            pbar = tqdm(trn_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_x, batch_y in pbar:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.type(torch.int64).to(self.device)
                
                # One-hot encode Y
                y_onehot = torch.zeros(batch_y.size(0), 2, device=self.device)
                y_onehot.scatter_(1, batch_y.unsqueeze(1), 1)
                
                M, Z = self.extract_features(batch_x)
                
                # Optimize mine_M
                self.opt_M.zero_grad()
                loss_M = self.mine_M(M, y_onehot)
                loss_M.backward()
                self.opt_M.step()
                
                # Optimize mine_Z
                self.opt_Z.zero_grad()
                loss_Z = self.mine_Z(Z, y_onehot)
                loss_Z.backward()
                self.opt_Z.step()
                
                loss_M_total += loss_M.item()
                loss_Z_total += loss_Z.item()
                steps += 1
                
                pbar.set_postfix({
                    "MI_M": f"{-loss_M.item():.4f}", 
                    "MI_Z": f"{-loss_Z.item():.4f}"
                })
                
            print(f"Epoch {epoch+1} Avg MI_M: {-loss_M_total/steps:.4f} Avg MI_Z: {-loss_Z_total/steps:.4f}")

def get_eval_loader_with_labels(config, batch_size):
    eval_trial_path = config["eval_trial_path"]
    eval_database_path = config["eval_database_path"]
    
    d_label_eval = {}
    file_eval = []
    track_info_eval = {}
    with open(eval_trial_path, "r") as f:
        l_meta = [line for line in f.readlines() if line.strip()]
        for line in l_meta:
            speaker_id, key, _, attack_type, label = line.strip().split(" ")
            file_eval.append(key)
            d_label_eval[key] = 1 if label == "bonafide" else 0
            track_info_eval[key] = {"speaker_id": speaker_id, "attack_type": attack_type}
            
    eval_set = Dataset_ASVspoof2019_train(
        list_IDs=file_eval,
        labels=d_label_eval,
        base_dir=eval_database_path,
        algo={"is_vsasv": config.get("dataset") == "VSASV", "track_info": track_info_eval},
        musan_prob=0.0
    )
    
    eval_loader = DataLoader(
        eval_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )
    return eval_loader

def main():
    parser = argparse.ArgumentParser(description="MINE Mutual Information Analysis")
    parser.add_argument("--config", type=str, default="./config/AASIST.conf", help="configuration file")
    parser.add_argument("--weights", type=str, default="./exp_result/LA_AASIST_ep100_bs64/weights/best.pth", help="pretrained model weights")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    analyzer = MINEAnalysis(args.config, args.weights, device, args.batch_size)
    
    train_generator = torch.Generator()
    train_generator.manual_seed(1234)
    set_seed(1234, analyzer.config)
    
    trn_list_path = analyzer.config["trn_list_path"]
    trn_database_path = analyzer.config["trn_database_path"]
    
    d_label_trn, file_train, track_info_trn = genSpoof_list(dir_meta=trn_list_path, is_train=True, is_eval=False)
    
    train_set = Dataset_ASVspoof2019_train(
        list_IDs=file_train,
        labels=d_label_trn,
        base_dir=trn_database_path,
        algo={"is_vsasv": analyzer.config.get("dataset") == "VSASV", "track_info": track_info_trn},
        musan_prob=0.0 
    )
    
    trn_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    
    analyzer.train(trn_loader, epochs=args.epochs)
    
    eval_loader = get_eval_loader_with_labels(analyzer.config, args.batch_size)
    
    analyzer.mine_M.eval()
    analyzer.mine_Z.eval()
    
    mi_M_total = 0.0
    mi_Z_total = 0.0
    steps = 0
    
    print("Evaluating MINE estimators...")
    with torch.no_grad():
        for batch_x, batch_y in tqdm(eval_loader, desc="Eval"):
            batch_x = batch_x.to(device)
            batch_y = batch_y.type(torch.int64).to(device)
            
            y_onehot = torch.zeros(batch_y.size(0), 2, device=device)
            y_onehot.scatter_(1, batch_y.unsqueeze(1), 1)
            
            M, Z = analyzer.extract_features(batch_x)
            
            # Using mi() method which correctly evaluates in no_grad
            # Or manually forward. MINE.forward returns loss.
            # mine_M(x, z, z_marg)
            # When evaluating MI on a held-out set, we still need z_marg to compute the denominator.
            # So we use the same forward method
            loss_M = analyzer.mine_M(M, y_onehot)
            loss_Z = analyzer.mine_Z(Z, y_onehot)
            
            mi_M_total += -loss_M.item()
            mi_Z_total += -loss_Z.item()
            steps += 1
            
    final_mi_M = mi_M_total / steps
    final_mi_Z = mi_Z_total / steps
    
    print("="*50)
    print("MUTUAL INFORMATION ESTIMATES")
    print(f"I(Y; M) = {final_mi_M:.4f}")
    print(f"I(Y; Z) = {final_mi_Z:.4f}")
    print(f"I(Y; Z) - I(Y; M) = {final_mi_Z - final_mi_M:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
