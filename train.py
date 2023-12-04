#%%
# %pip install transformer_lens==1.2.1
# %pip install git+https://github.com/neelnanda-io/neel-plotly
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path
from argparse import ArgumentParser
import pprint
import argparse
import wandb
import tqdm
from datasets import load_dataset
import einops
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from functools import partial
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR
#%%
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
SAVE_DIR = Path("/workspace")

#%%

default_cfg = {
    "seed": 49,
    "batch_size": 1024,
    "buffer_mult": 384,
    "lr": 3e-4,
    "num_tokens": int(2e9),
    "l1_coeff": 1e-4,
    "beta1": 0.9,
    "beta2": 0.99,
    "dict_mult": 32,
    "seq_len": 128,
    "enc_dtype":"fp32",
    "remove_rare_dir": False,
    "model_name": "gpt2-small",
    "site": "mlp_out",
    "layer": 0,
    "device": "cuda:0"
}
site_to_size = {
    "mlp_out": 768,
    "post": 2048,
    "resid_pre": 768,
    "resid_mid": 768,
    "resid_post": 768,
}

def post_init_cfg(cfg):
    cfg["model_batch_size"] = cfg["batch_size"] // cfg["seq_len"] * 16
    cfg["buffer_size"] = cfg["batch_size"] * cfg["buffer_mult"]
    cfg["buffer_batches"] = cfg["buffer_size"] // cfg["seq_len"]
    cfg["act_size"] = site_to_size[cfg["site"]]
    cfg["act_name"] = utils.get_act_name(cfg["site"], cfg["layer"])
    cfg["dict_size"] = cfg["act_size"] * cfg["dict_mult"]
    cfg["name"] = f"{cfg['model_name']}_{cfg['layer']}_{cfg['dict_size']}_{cfg['site']}"
post_init_cfg(default_cfg)
cfg = default_cfg
#%%
model = HookedTransformer.from_pretrained(cfg["model_name"]).to(DTYPES[cfg["enc_dtype"]]).to(cfg["device"])

#%%
def shuffle_data(all_tokens):
    print("Shuffled data")
    return all_tokens[torch.randperm(all_tokens.shape[0])]

loading_data_first_time = False
if loading_data_first_time:
    data = load_dataset("NeelNanda/c4-code-tokenized-2b", split="train", cache_dir="/workspace/cache/")
    data.save_to_disk("/workspace/data/c4_code_tokenized_2b.hf")
    data.set_format(type="torch", columns=["tokens"])
    all_tokens = data["tokens"]
    all_tokens.shape


    all_tokens_reshaped = einops.rearrange(all_tokens, "batch (x seq_len) -> (batch x) seq_len", x=8, seq_len=128)
    all_tokens_reshaped[:, 0] = model.tokenizer.bos_token_id
    all_tokens_reshaped = all_tokens_reshaped[torch.randperm(all_tokens_reshaped.shape[0])]
    torch.save(all_tokens_reshaped, "/workspace/data/c4_code_2b_tokens_reshaped.pt")
else:
    # data = datasets.load_from_disk("/workspace/data/c4_code_tokenized_2b.hf")
    all_tokens = torch.load("/workspace/data/c4_code_2b_tokens_reshaped.pt")
    all_tokens = shuffle_data(all_tokens)
#%%
class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_hidden = cfg["dict_size"]
        l1_coeff = cfg["l1_coeff"]
        dtype = DTYPES[cfg["enc_dtype"]]
        torch.manual_seed(cfg["seed"])
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg["act_size"], d_hidden, dtype=dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, cfg["act_size"], dtype=dtype)))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(cfg["act_size"], dtype=dtype))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff

        self.to(cfg["device"])
    
    def forward(self, x, fraction=1):

        norm = torch.linalg.vector_norm(x, dim=-1, keepdims=True)
        x_norm = x / norm
        x_cent = x_norm - self.b_dec

        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct_norm = acts @ self.W_dec + self.b_dec

        l2_loss = (x_reconstruct_norm.float() - x_norm.float()).pow(2).sum(-1).mean(0)
        x_reconstruct = x_reconstruct_norm * norm

        # squared_diff = (x_reconstruct.float() - x.float()).pow(2)
        # squared_norm = x.float().pow(2).sum(-1, keepdim=True)
        # scaled_squared_diff = squared_diff / squared_norm
        # l2_loss = scaled_squared_diff.sum(-1).mean(0)

        current_l1_coefficient = self.get_l1_coeff(fraction)
        l1_loss = current_l1_coefficient * (acts.float().abs().sum())
        l0_norm = (acts > 0).sum() / acts.shape[0]
        loss = l2_loss + l1_loss
        return loss, x_reconstruct, acts, l2_loss, l1_loss, l0_norm
    
    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        # Bugfix(?) for ensuring W_dec retains unit norm, this was not there when I trained my original autoencoders.
        self.W_dec.data = W_dec_normed
    
    def get_version(self):
        version_list = [int(file.name.split(".")[0]) for file in list(SAVE_DIR.iterdir()) if "pt" in str(file)]
        if len(version_list):
            return 1+max(version_list)
        else:
            return 0

    def save(self):
        version = self.get_version()
        torch.save(self.state_dict(), SAVE_DIR/(str(version)+".pt"))
        with open(SAVE_DIR/(str(version)+"_cfg.json"), "w") as f:
            json.dump(cfg, f)
        print("Saved as version", version)
    
    @classmethod
    def load(cls, version):
        cfg = (json.load(open(SAVE_DIR/(str(version)+"_cfg.json"), "r")))
        pprint.pprint(cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(torch.load(SAVE_DIR/(str(version)+".pt")))
        return self

    def get_l1_coeff(self, current_frac, warmup_frac=0.05):
        if current_frac < warmup_frac:
            return self.l1_coeff * (current_frac / warmup_frac)
        else:
            return self.l1_coeff

class Buffer():
    """
    This defines a data buffer, to store a bunch of MLP acts that can be used to train the autoencoder. It'll automatically run the model to generate more when it gets halfway empty. 
    """
    def __init__(self, cfg):
        self.buffer = torch.zeros((cfg["buffer_size"], cfg["act_size"]), dtype=torch.bfloat16, requires_grad=False).to(cfg["device"])
        self.cfg = cfg
        self.token_pointer = 0
        self.first = True
        self.refresh()
    
    @torch.no_grad()
    def refresh(self):
        self.pointer = 0
        with torch.autocast("cuda", torch.bfloat16):
            if self.first:
                num_batches = self.cfg["buffer_batches"]
            else:
                num_batches = self.cfg["buffer_batches"]//2
            self.first = False
            for _ in range(0, num_batches, self.cfg["model_batch_size"]):
                tokens = all_tokens[self.token_pointer:self.token_pointer+self.cfg["model_batch_size"]]
                layer = np.random.randint(12)
                site = np.random.choice(["resid_pre", "resid_mid", "resid_post"])
                act_name = utils.get_act_name(site, layer)
                _, cache = model.run_with_cache(tokens, stop_at_layer=layer+1)
                acts = cache[act_name].reshape(-1, self.cfg["act_size"])
                if self.pointer < cfg["buffer_size"]:
                    self.buffer[self.pointer: self.pointer+acts.shape[0]] = acts
                    self.pointer += acts.shape[0]
                    self.token_pointer += self.cfg["model_batch_size"]
                    # if self.token_pointer > all_tokens.shape[0] - self.cfg["model_batch_size"]:
                    #     self.token_pointer = 0

        self.pointer = 0
        self.buffer = self.buffer[torch.randperm(self.buffer.shape[0]).to(cfg["device"])]

    @torch.no_grad()
    def next(self):
        out = self.buffer[self.pointer:self.pointer+self.cfg["batch_size"]]
        self.pointer += self.cfg["batch_size"]
        if self.pointer > self.buffer.shape[0]//2 - self.cfg["batch_size"]:
            # print("Refreshing the buffer!")
            self.refresh()
        return out

# %%
encoder = AutoEncoder(cfg)
buffer = Buffer(cfg)
# %%
@torch.no_grad()
def get_recons_loss(num_batches=5, local_encoder=None):
    if local_encoder is None:
        local_encoder = encoder
    loss_list = []
    for layer in tqdm.trange(12):
        for site in ["resid_pre", "resid_mid", "resid_post"]:
            act_name = utils.get_act_name(site, layer)

            tokens = all_tokens[torch.randperm(len(all_tokens))[:cfg["model_batch_size"]]]
            loss = model(tokens, return_type="loss")
            recons_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(act_name, partial(replacement_hook, encoder=local_encoder))])
            mean_abl_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(act_name, mean_ablate_hook)])
            zero_abl_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(act_name, zero_ablate_hook)])
            loss_list.append((loss, recons_loss, zero_abl_loss, mean_abl_loss))
    losses = torch.tensor(loss_list)
    loss, recons_loss, zero_abl_loss, mean_abl_loss = losses.mean(0).tolist()

    print(loss, recons_loss, zero_abl_loss, mean_abl_loss)
    score = ((zero_abl_loss - recons_loss)/(zero_abl_loss - loss))
    score2 = ((mean_abl_loss - recons_loss)/(mean_abl_loss - loss))
    print(f"zero_score {score:.2%}")
    print(f"mean_score {score2:.2%}")

    # print(f"{((zero_abl_loss - mean_abl_loss)/(zero_abl_loss - loss)).item():.2%}")
    return score, loss, recons_loss, zero_abl_loss, score2

# Frequency
@torch.no_grad()
def get_freqs(num_batches=25, local_encoder=None):
    if local_encoder is None:
        local_encoder = encoder
    act_freq_scores = torch.zeros(local_encoder.d_hidden, dtype=torch.float32).to(cfg["device"])
    total = 0
    for i in tqdm.trange(num_batches):
        tokens = all_tokens[torch.randperm(len(all_tokens))[:cfg["model_batch_size"]]]
        layer = np.random.randint(12)
        site = np.random.choice(["resid_pre", "resid_mid", "resid_post"])
        act_name = utils.get_act_name(site, layer)
        _, cache = model.run_with_cache(tokens, stop_at_layer=layer+1)
        acts = cache[act_name]
        acts = acts.reshape(-1, cfg["act_size"])

        hidden = local_encoder(acts)[2]
        
        act_freq_scores += (hidden > 0).sum(0)
        total+=hidden.shape[0]
    act_freq_scores /= total
    num_dead = (act_freq_scores==0).float().mean()
    print("Num dead", num_dead)
    return act_freq_scores

@torch.no_grad()
def re_init(indices, encoder):
    new_W_enc = (torch.nn.init.kaiming_uniform_(torch.zeros_like(encoder.W_enc)))
    new_W_dec = (torch.nn.init.kaiming_uniform_(torch.zeros_like(encoder.W_dec)))
    new_b_enc = (torch.zeros_like(encoder.b_enc))
    print(new_W_dec.shape, new_W_enc.shape, new_b_enc.shape)
    encoder.W_enc.data[:, indices] = new_W_enc[:, indices]
    encoder.W_dec.data[indices, :] = new_W_dec[indices, :]
    encoder.b_enc.data[indices] = new_b_enc[indices]
#%%
def replacement_hook(mlp_post, hook, encoder):
    mlp_post_reconstr = encoder(mlp_post)[1]
    return mlp_post_reconstr

def mean_ablate_hook(mlp_post, hook):
    mlp_post[:] = mlp_post.mean([0, 1])
    return mlp_post

def zero_ablate_hook(mlp_post, hook):
    mlp_post[:] = 0.
    return mlp_post

#%%
# !wandb login b996b5b2faffea971ac27f3de099ffb0a1c98ee9
#%%
try:
    wandb.init(project="autoencoders", entity="hiibb", config=cfg)
    num_batches = cfg["num_tokens"] // cfg["batch_size"]
    # model_num_batches = cfg["model_batch_size"] * num_batches
    encoder_optim = torch.optim.Adam(encoder.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    scheduler = OneCycleLR(encoder_optim, max_lr=cfg["lr"], total_steps=num_batches)
    recons_scores = []
    act_freq_scores_list = []
    counter = 0
    for i in tqdm.trange(num_batches):
        counter += 1
        i = i % all_tokens.shape[0]
        acts = buffer.next()
        loss, x_reconstruct, mid_acts, l2_loss, l1_loss, l0_norm = encoder(acts, fraction=counter/num_batches)
        loss.backward()
        encoder.make_decoder_weights_and_grad_unit_norm()
        encoder_optim.step()
        scheduler.step()
        encoder_optim.zero_grad()
        loss_dict = {"loss": loss.item(), "l2_loss": l2_loss.item(), "l1_loss": l1_loss.item(), "l0_norm": l0_norm.item()}
        del loss, x_reconstruct, mid_acts, l2_loss, l1_loss, acts, l0_norm
        if (i) % 100 == 0:
            wandb.log(loss_dict)
        if (i) % 10000 == 9999:

            freqs = get_freqs(25, local_encoder=encoder)
            act_freq_scores_list.append(freqs)
            # histogram(freqs.log10(), marginal="box", histnorm="percent", title="Frequencies")
            wandb.log({

                "learning_rate": scheduler.get_last_lr()[0],
                "dead": (freqs==0).float().mean().item(),
                "below_1e-6": (freqs<1e-6).float().mean().item(),
                "below_1e-5": (freqs<1e-5).float().mean().item(),
                "below_1e-4": (freqs<1e-4).float().mean().item(),
                "below_1e-3": (freqs<1e-3).float().mean().item(),
                "below_1e-2": (freqs<1e-2).float().mean().item(),
                "below_1e-1": (freqs<1e-1).float().mean().item(),            
            })
        if (i+1) % 50000 == 0:
            x = (get_recons_loss(local_encoder=encoder, num_batches=50))
            print("Reconstruction:", x)

            encoder.save()
            freqs = get_freqs(100, local_encoder=encoder)
            to_be_reset = (freqs<10**(-5.5))
            wandb.log({"reset_neurons": to_be_reset.sum(),
                "recons_score_zero": x[0],
                "recons_score_mean": x[-1]})
            print("Resetting neurons!", to_be_reset.sum())
            re_init(to_be_reset, encoder)
finally:
    encoder.save()
# %%
torch.cuda.empty_cache()
# %%
