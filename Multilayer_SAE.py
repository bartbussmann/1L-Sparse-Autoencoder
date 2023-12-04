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
from collections import namedtuple


#%%
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
SAVE_DIR = Path("/workspace")
loading_data_first_time = False


#%%

default_cfg = {
    "seed": 49,
    "batch_size": 4096,
    "buffer_mult": 384,
    "lr": 3e-5,
    "num_tokens": int(2e9),
    "l1_coeff": 2e-3,
    "l1_coeff_W": 2e-3,
    "beta1": 0.9,
    "beta2": 0.99,
    "dict_mult": 32,
    "seq_len": 128,
    "enc_dtype":"fp32",
    "remove_rare_dir": False,
    "model_name": "gelu-1l",
    "site": "mlp_out",
    "layer": 0,
    "device": "cuda:0"
}
site_to_size = {
    "mlp_out": 512,
    "post": 2048,
    "resid_pre": 512,
    "resid_mid": 512,
    "resid_post": 512,
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
        d_hidden1 = cfg["dict_size"]
        d_hidden2 = cfg["dict_size"] // 4  
        l1_coeff = cfg["l1_coeff"]
        l1_coeff_W = cfg["l1_coeff_W"]
        dtype = DTYPES[cfg["enc_dtype"]]
        torch.manual_seed(cfg["seed"])
        self.W_enc1 = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg["act_size"], d_hidden1, dtype=dtype)))
        self.W_enc2 = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_hidden1, d_hidden2, dtype=dtype)))
        self.W_dec1 = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_hidden1, cfg["act_size"], dtype=dtype)))
        self.W_dec2 = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_hidden2, cfg["act_size"], dtype=dtype)))
        self.b_enc1 = nn.Parameter(torch.zeros(d_hidden1, dtype=dtype))
        self.b_enc2 = nn.Parameter(torch.zeros(d_hidden2, dtype=dtype))
        self.b_dec1 = nn.Parameter(torch.zeros(cfg["act_size"], dtype=dtype))
        self.b_dec2 = nn.Parameter(torch.zeros(cfg["act_size"], dtype=dtype))

        self.W_dec1.data[:] = self.W_dec1 / self.W_dec1.norm(dim=-1, keepdim=True)
        self.W_dec2.data[:] = self.W_dec2 / self.W_dec2.norm(dim=-1, keepdim=True)

        self.d_hidden1 = d_hidden1
        self.d_hidden2 = d_hidden2
        self.l1_coeff = l1_coeff
        self.l1_coeff_W = l1_coeff_W

        self.to(cfg["device"])
    
    def forward(self, x, fraction=1):
        x_cent = x - ((self.b_dec1 + self.b_dec2) / 2)
        acts1 = F.relu(x_cent @ self.W_enc1 + self.b_enc1)
        acts2 = F.relu(acts1 @ self.W_enc2 + self.b_enc2)
        x_reconstruct = acts1 @ self.W_dec1 + self.b_dec1
        x_reconstruct2 = acts2 @ self.W_dec2 + self.b_dec2
        l2_loss_1 =  (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
        l2_loss_2 = (x_reconstruct2.float() - x.float()).pow(2).sum(-1).mean(0)
        l2_loss = (l2_loss_1 + l2_loss_2) / 2
        l1_loss_1 = self.l1_coeff * (acts1.float().abs().sum()) * fraction
        l1_loss_2 = self.l1_coeff * (acts2.float().abs().sum()) * fraction
        l1_loss = (l1_loss_1 + l1_loss_2) / 2
        l1_loss_W = self.l1_coeff_W * self.W_enc2.abs().sum() * fraction 
        l0_norm_1 = (acts1 > 0).sum() / acts1.shape[0]
        l0_norm_2 = (acts2 > 0).sum() / acts2.shape[0]
        l0_norm = (l0_norm_1 + l0_norm_2) / 2
        loss = l2_loss + l1_loss + l1_loss_W  
        LossOutput = namedtuple('LossOutput', ['loss', 'x_reconstruct', 'acts1', 'l2_loss', 'l1_loss', 'l0_norm', 'acts2', 'l1_loss_W', 'x_reconstruct2', 'l0_norm_1', 'l0_norm_2'])
        return LossOutput(loss, x_reconstruct, acts1, l2_loss, l1_loss, l0_norm, acts2, l1_loss_W, x_reconstruct2, l0_norm_1, l0_norm_2)
    
    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec1 / self.W_dec1.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec1.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec1.grad -= W_dec_grad_proj
        self.W_dec1.data = W_dec_normed
        W_dec2_normed = self.W_dec2 / self.W_dec2.norm(dim=-1, keepdim=True)
        W_dec2_grad_proj = (self.W_dec2.grad * W_dec2_normed).sum(-1, keepdim=True) * W_dec2_normed
        self.W_dec2.grad -= W_dec2_grad_proj
        self.W_dec2.data = W_dec2_normed
    
    def get_version(self):
        version_list = [int(file.name.split(".")[0]) for file in list(SAVE_DIR.iterdir()) if "pt" in str(file)]
        if len(version_list):
            return 1+max(version_list)
        else:
            return 0

    def save(self, reconstr, l0):
        version = self.get_version()
        torch.save(self.state_dict(), SAVE_DIR/(str(version)+".pt"))
        cfg["reconstruction_loss"] = reconstr
        cfg["l0"] = l0
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
                _, cache = model.run_with_cache(tokens, stop_at_layer=cfg["layer"]+1)
                acts = cache[cfg["act_name"]].reshape(-1, self.cfg["act_size"])
                # print(tokens.shape, acts.shape, self.pointer, self.token_pointer)
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
#%%
# %%
encoder = AutoEncoder(cfg)
buffer = Buffer(cfg)
# %%
def mean_ablate_hook(mlp_post, hook):
    mlp_post[:] = mlp_post.mean([0, 1])
    return mlp_post

def zero_ablate_hook(mlp_post, hook):
    mlp_post[:] = 0.
    return mlp_post

def replacement_hook(mlp_post, hook, encoder, use_second_reconstruct=False):
    if use_second_reconstruct:
        mlp_post_reconstr = encoder(mlp_post).x_reconstruct2
    else:
        mlp_post_reconstr = encoder(mlp_post).x_reconstruct
    return mlp_post_reconstr

@torch.no_grad()
def get_recons_loss(num_batches=5, local_encoder=None, use_second_reconstruct=False):
    if local_encoder is None:
        local_encoder = encoder
    loss_list = []
    for i in range(num_batches):
        tokens = all_tokens[torch.randperm(len(all_tokens))[:cfg["model_batch_size"]]]
        loss = model(tokens, return_type="loss")
        recons_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(cfg["act_name"], partial(replacement_hook, encoder=local_encoder, use_second_reconstruct=use_second_reconstruct))])
        # mean_abl_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(cfg["act_name"], mean_ablate_hook)])
        zero_abl_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(cfg["act_name"], zero_ablate_hook)])
        loss_list.append((loss, recons_loss, zero_abl_loss))
    losses = torch.tensor(loss_list)
    loss, recons_loss, zero_abl_loss = losses.mean(0).tolist()

    print(loss, recons_loss, zero_abl_loss)
    score = ((zero_abl_loss - recons_loss)/(zero_abl_loss - loss))
    print(f"{score:.2%}")
    # print(f"{((zero_abl_loss - mean_abl_loss)/(zero_abl_loss - loss)).item():.2%}")
    return score, loss, recons_loss, zero_abl_loss

# Frequency
@torch.no_grad()
def get_freqs(num_batches=25, local_encoder=None):
    if local_encoder is None:
        local_encoder = encoder
    act_freq_scores = torch.zeros(local_encoder.d_hidden1, dtype=torch.float32).to(cfg["device"])
    total = 0
    for i in tqdm.trange(num_batches):
        tokens = all_tokens[torch.randperm(len(all_tokens))[:cfg["model_batch_size"]]]
        
        _, cache = model.run_with_cache(tokens, stop_at_layer=cfg["layer"]+1)
        acts = cache[cfg["act_name"]]
        acts = acts.reshape(-1, cfg["act_size"])

        hidden = local_encoder(acts).acts1
        
        act_freq_scores += (hidden > 0).sum(0)
        total+=hidden.shape[0]
    act_freq_scores /= total
    num_dead = (act_freq_scores==0).float().mean()
    print("Num dead", num_dead)
    return act_freq_scores

@torch.no_grad()
def re_init(indices, encoder):
    new_W_enc = (torch.nn.init.kaiming_uniform_(torch.zeros_like(encoder.W_enc1)))
    new_W_dec = (torch.nn.init.kaiming_uniform_(torch.zeros_like(encoder.W_dec1)))
    new_b_enc = (torch.zeros_like(encoder.b_enc1))
    print(new_W_dec.shape, new_W_enc.shape, new_b_enc.shape)
    encoder.W_enc1.data[:, indices] = new_W_enc[:, indices]
    encoder.W_dec1.data[indices, :] = new_W_dec[indices, :]
    encoder.b_enc1.data[indices] = new_b_enc[indices]
#%%


#%%
# !wandb login b996b5b2faffea971ac27f3de099ffb0a1c98ee9
#%%
wandb.init(project="autoencoders", entity="hiibb")
num_batches = cfg["num_tokens"] // cfg["batch_size"]
# model_num_batches = cfg["model_batch_size"] * num_batches
encoder_optim = torch.optim.Adam(encoder.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
recons_scores = []
act_freq_scores_list = []
counter = 0
for i in tqdm.trange(num_batches):
    counter += 1
    i = i % all_tokens.shape[0]
    acts = buffer.next()
    loss_output = encoder(acts, fraction=counter/num_batches)
    loss_output.loss.backward()
    encoder.make_decoder_weights_and_grad_unit_norm()
    encoder_optim.step()
    encoder_optim.zero_grad()
    loss_dict = {"loss": loss_output.loss.item(), "l2_loss": loss_output.l2_loss.item(), "l1_loss": loss_output.l1_loss.item(), "l1_loss_W": loss_output.l1_loss_W.item(), "l0_norm_1": loss_output.l0_norm_1.item(), "l0_norm_2": loss_output.l0_norm_2.item()}
    del loss_output
    if (i) % 100 == 0:
        wandb.log(loss_dict)
        print(loss_dict)
    if (i) % 1000 == 0:
        x = (get_recons_loss(local_encoder=encoder))
        recons_score1 = x[0]
        x = (get_recons_loss(local_encoder=encoder, use_second_reconstruct=True))
        recons_score2 = x[0]
        print("Reconstruction:", recons_score1, recons_score2)
        recons_scores.append(recons_score1)
        freqs = get_freqs(5, local_encoder=encoder)
        act_freq_scores_list.append(freqs)
        n_large_weights = (encoder.W_enc2.abs() > 1e-4).float().sum().item()

        # histogram(freqs.log10(), marginal="box", histnorm="percent", title="Frequencies")
        wandb.log({
            "recons_score1": recons_score1,
            "recons_score2": recons_score2,    
            "dead": (freqs==0).float().mean().item(),
            "below_1e-6": (freqs<1e-6).float().mean().item(),
            "below_1e-5": (freqs<1e-5).float().mean().item(),
            "below_1e-4": (freqs<1e-4).float().mean().item(),
            "below_1e-3": (freqs<1e-3).float().mean().item(),
            "below_1e-2": (freqs<1e-2).float().mean().item(),
            "below_1e-1": (freqs<1e-1).float().mean().item(),
            "n_large_weights": n_large_weights

        })
    if (i+1) % 30000 == 0:
        x = (get_recons_loss(local_encoder=encoder))
        acts = buffer.next()
        loss_output = encoder(acts, fraction=counter/num_batches)
        encoder.save(x[0], loss_output.l0_norm.item())
        freqs = get_freqs(50, local_encoder=encoder)
        to_be_reset = (freqs<10**(-5.5))
        wandb.log({"reset_neurons": to_be_reset.sum()})

        print("Resetting neurons!", to_be_reset.sum())
        re_init(to_be_reset, encoder)
# finally:
#     encoder.save()
# %%
torch.cuda.empty_cache()
# %%
