# -*- coding: utf-8 -*-
"""Autoencoder GPT2-small.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qbEap1RWLusV7vVa-K00UCU24uv40ekg
"""

!python3 -V

"""# Setup

## Dependencies
"""

!pip install transformer_lens
!pip install gradio

import transformer_lens
from transformer_lens import HookedTransformer, utils
import torch
import numpy as np
import gradio as gr
import pprint
import json
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import HfApi
from IPython.display import HTML
from functools import partial
import tqdm.notebook as tqdm
import plotly.express as px
import pandas as pd

"""## Defining the Autoencoder"""

cfg = {
    "seed": 49,
    "batch_size": 4096,
    "buffer_mult": 384,
    "lr": 1e-4,
    "num_tokens": int(2e9),
    "l1_coeff": 3e-4,
    "beta1": 0.9,
    "beta2": 0.99,
    "dict_mult": 32,
    "seq_len": 128,
    "d_mlp": 2048,
    "enc_dtype":"fp32",
    "remove_rare_dir": False,
}
cfg["model_batch_size"] = 64
cfg["buffer_size"] = cfg["batch_size"] * cfg["buffer_mult"]
cfg["buffer_batches"] = cfg["buffer_size"] // cfg["seq_len"]

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
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
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
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

    def get_l1_coeff(self, current_frac, warmup_frac=0.5):
        if current_frac < warmup_frac:
            return self.l1_coeff * (current_frac / warmup_frac)
        else:
            return self.l1_coeff

"""## Utils

### Get Reconstruction Loss
"""

def replacement_hook(mlp_post, hook, encoder):
    mlp_post_reconstr = encoder(mlp_post)[1]
    return mlp_post_reconstr

def mean_ablate_hook(mlp_post, hook):
    mlp_post[:] = mlp_post.mean([0, 1])
    return mlp_post

def zero_ablate_hook(mlp_post, hook):
    mlp_post[:] = 0.
    return mlp_post

@torch.no_grad()
def get_recons_loss(num_batches=5, local_encoder=None, act_name=None):
    if local_encoder is None:
        local_encoder = encoder
    if act_name is None:
        act_name = cfg["act_name"]
    loss_list = []
    for i in range(num_batches):
        tokens = all_tokens[torch.randperm(len(all_tokens))[:cfg["model_batch_size"]]]
        loss = model(tokens, return_type="loss")
        recons_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(act_name, partial(replacement_hook, encoder=local_encoder))])
        # mean_abl_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(act_name, mean_ablate_hook)])
        zero_abl_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(act_name, zero_ablate_hook)])
        loss_list.append((loss, recons_loss, zero_abl_loss))
    losses = torch.tensor(loss_list)
    loss, recons_loss, zero_abl_loss = losses.mean(0).tolist()

    print(loss, recons_loss, zero_abl_loss)
    score = ((zero_abl_loss - recons_loss)/(zero_abl_loss - loss))
    print(f"{score:.2%}")
    # print(f"{((zero_abl_loss - mean_abl_loss)/(zero_abl_loss - loss)).item():.2%}")
    return score, loss, recons_loss, zero_abl_loss

"""### Get Frequencies"""

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

        hidden = local_encoder(acts)[2]

        act_freq_scores += (hidden > 0).sum(0)
        total+=hidden.shape[0]
    act_freq_scores /= total
    num_dead = (act_freq_scores==0).float().mean()
    print("Num dead", num_dead)
    return act_freq_scores

"""## Visualise Feature Utils"""

from html import escape
import colorsys

from IPython.display import display

SPACE = "·"
NEWLINE="↩"
TAB = "→"

def create_html(strings, values, max_value=None, saturation=0.5, allow_different_length=False, return_string=False):
    # escape strings to deal with tabs, newlines, etc.
    escaped_strings = [escape(s, quote=True) for s in strings]
    processed_strings = [
        s.replace("\n", f"{NEWLINE}<br/>").replace("\t", f"{TAB}&emsp;").replace(" ", "&nbsp;")
        for s in escaped_strings
    ]

    if isinstance(values, torch.Tensor) and len(values.shape)>1:
        values = values.flatten().tolist()

    if not allow_different_length:
        assert len(processed_strings) == len(values)

    # scale values
    if max_value is None:
        max_value = max(max(values), -min(values))+1e-3
    scaled_values = [v / max_value * saturation for v in values]

    # create html
    html = ""
    for i, s in enumerate(processed_strings):
        if i<len(scaled_values):
            v = scaled_values[i]
        else:
            v = 0
        if v < 0:
            hue = 0  # hue for red in HSV
        else:
            hue = 0.66  # hue for blue in HSV
        rgb_color = colorsys.hsv_to_rgb(
            hue, v, 1
        )  # hsv color with hue 0.66 (blue), saturation as v, value 1
        hex_color = "#%02x%02x%02x" % (
            int(rgb_color[0] * 255),
            int(rgb_color[1] * 255),
            int(rgb_color[2] * 255),
        )
        html += f'<span style="background-color: {hex_color}; border: 1px solid lightgray; font-size: 16px; border-radius: 3px;">{s}</span>'
    if return_string:
        return html
    else:
        display(HTML(html))

def basic_feature_vis(text, feature_index, max_val=0):
    feature_in = encoder.W_enc[:, feature_index]
    feature_bias = encoder.b_enc[feature_index]
    _, cache = model.run_with_cache(text, stop_at_layer=1, names_filter=utils.get_act_name("post", 0))
    mlp_acts = cache[utils.get_act_name("post", 0)][0]
    feature_acts = F.relu((mlp_acts - encoder.b_dec) @ feature_in + feature_bias)
    if max_val==0:
        max_val = max(1e-7, feature_acts.max().item())
        # print(max_val)
    # if min_val==0:
    #     min_val = min(-1e-7, feature_acts.min().item())
    return basic_token_vis_make_str(text, feature_acts, max_val)
def basic_token_vis_make_str(strings, values, max_val=None):
    if not isinstance(strings, list):
        strings = model.to_str_tokens(strings)
    values = utils.to_numpy(values)
    if max_val is None:
        max_val = values.max()
    # if min_val is None:
    #     min_val = values.min()
    header_string = f"<h4>Max Range <b>{values.max():.4f}</b> Min Range: <b>{values.min():.4f}</b></h4>"
    header_string += f"<h4>Set Max Range <b>{max_val:.4f}</b></h4>"
    # values[values>0] = values[values>0]/ma|x_val
    # values[values<0] = values[values<0]/abs(min_val)
    body_string = create_html(strings, values, max_value=max_val, return_string=True)
    return header_string + body_string
# display(HTML(basic_token_vis_make_str(tokens[0, :10], mlp_acts[0, :10, 7], 0.1)))
# # %%
# The `with gr.Blocks() as demo:` syntax just creates a variable called demo containing all these components
import gradio as gr
try:
    demos[0].close()
except:
    pass
demos = [None]
def make_feature_vis_gradio(feature_id, starting_text=None, batch=None, pos=None):
    if starting_text is None:
        starting_text = model.to_string(all_tokens[batch, 1:pos+1])
    try:
        demos[0].close()
    except:
        pass
    with gr.Blocks() as demo:
        gr.HTML(value=f"Hacky Interactive Neuroscope for gelu-1l")
        # The input elements
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Text", value=starting_text)
                # Precision=0 makes it an int, otherwise it's a float
                # Value sets the initial default value
                feature_index = gr.Number(
                    label="Feature Index", value=feature_id, precision=0
                )
                # # If empty, these two map to None
                max_val = gr.Number(label="Max Value", value=None)
                # min_val = gr.Number(label="Min Value", value=None)
                inputs = [text, feature_index, max_val]
        with gr.Row():
            with gr.Column():
                # The output element
                out = gr.HTML(label="Neuron Acts", value=basic_feature_vis(starting_text, feature_id))
        for inp in inputs:
            inp.change(basic_feature_vis, inputs, out)
    demo.launch(share=True)
    demos[0] = demo

"""### Inspecting Top Logits"""

SPACE = "·"
NEWLINE="↩"
TAB = "→"
def process_token(s):
    if isinstance(s, torch.Tensor):
        s = s.item()
    if isinstance(s, np.int64):
        s = s.item()
    if isinstance(s, int):
        s = model.to_string(s)
    s = s.replace(" ", SPACE)
    s = s.replace("\n", NEWLINE+"\n")
    s = s.replace("\t", TAB)
    return s

def process_tokens(l):
    if isinstance(l, str):
        l = model.to_str_tokens(l)
    elif isinstance(l, torch.Tensor) and len(l.shape)>1:
        l = l.squeeze(0)
    return [process_token(s) for s in l]

def process_tokens_index(l):
    if isinstance(l, str):
        l = model.to_str_tokens(l)
    elif isinstance(l, torch.Tensor) and len(l.shape)>1:
        l = l.squeeze(0)
    return [f"{process_token(s)}/{i}" for i,s in enumerate(l)]

def create_vocab_df(logit_vec, make_probs=False, full_vocab=None):
    if full_vocab is None:
        full_vocab = process_tokens(model.to_str_tokens(torch.arange(model.cfg.d_vocab)))
    vocab_df = pd.DataFrame({"token": full_vocab, "logit": utils.to_numpy(logit_vec)})
    if make_probs:
        vocab_df["log_prob"] = utils.to_numpy(logit_vec.log_softmax(dim=-1))
        vocab_df["prob"] = utils.to_numpy(logit_vec.softmax(dim=-1))
    return vocab_df.sort_values("logit", ascending=False)

"""### Make Token DataFrame"""

def list_flatten(nested_list):
    return [x for y in nested_list for x in y]
def make_token_df(tokens, len_prefix=5, len_suffix=1):
    str_tokens = [process_tokens(model.to_str_tokens(t)) for t in tokens]
    unique_token = [[f"{s}/{i}" for i, s in enumerate(str_tok)] for str_tok in str_tokens]

    context = []
    batch = []
    pos = []
    label = []
    for b in range(tokens.shape[0]):
        # context.append([])
        # batch.append([])
        # pos.append([])
        # label.append([])
        for p in range(tokens.shape[1]):
            prefix = "".join(str_tokens[b][max(0, p-len_prefix):p])
            if p==tokens.shape[1]-1:
                suffix = ""
            else:
                suffix = "".join(str_tokens[b][p+1:min(tokens.shape[1]-1, p+1+len_suffix)])
            current = str_tokens[b][p]
            context.append(f"{prefix}|{current}|{suffix}")
            batch.append(b)
            pos.append(p)
            label.append(f"{b}/{p}")
    # print(len(batch), len(pos), len(context), len(label))
    return pd.DataFrame(dict(
        str_tokens=list_flatten(str_tokens),
        unique_token=list_flatten(unique_token),
        context=context,
        batch=batch,
        pos=pos,
        label=label,
    ))

"""## Loading the Model"""

model = HookedTransformer.from_pretrained("gpt2-small").to(DTYPES[cfg["enc_dtype"]])
n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab

"""## Loading Data"""

data = load_dataset("NeelNanda/c4-code-20k", split="train")
tokenized_data = utils.tokenize_and_concatenate(data, model.tokenizer, max_length=128)
tokenized_data = tokenized_data.shuffle(42)
all_tokens = tokenized_data["tokens"]

"""# Analysis

## Loading the Autoencoder

There are two runs on separate random seeds, along with a bunch of intermediate checkpoints
"""

# auto_encoder_run = "run1" # @param ["run1", "run2"]
version = 99
cfg = json.load(open(str(version) + "_cfg.json", "r"))

print(cfg)
encoder = AutoEncoder(cfg)
encoder.load_state_dict(torch.load(str(version)+".pt"))

"""## Using the Autoencoder

We run the model and replace the MLP activations with those reconstructed from the autoencoder, and get 91% loss recovered
"""

layer = np.random.randint(12)
site = np.random.choice(["resid_pre", "resid_mid", "resid_post"])

for layer in range(12):
    for site in ["resid_pre", "resid_mid", "resid_post"]:
        act_name = utils.get_act_name(site, layer)
        print(act_name)
        _ = get_recons_loss(num_batches=5, local_encoder=encoder, act_name=act_name)

"""## Rare Features Are All The Same

For each feature we can get the frequency at which it's non-zero (per token, averaged across a bunch of batches), and plot a histogram
"""

freqs = get_freqs(num_batches = 50, local_encoder = encoder)

# Add 1e-6.5 so that dead features show up as log_freq -6.5
log_freq = (freqs + 10**-6.5).log10()
px.histogram(utils.to_numpy(log_freq), title="Log Frequency of Features", histnorm='percent')

"""We see that it's clearly bimodal! Let's define rare features as those with freq < 1e-4, and look at the cosine sim of each feature with the average rare feature - we see that almost all rare features correspond to this feature!"""

is_rare = freqs < 1e-4
rare_enc = encoder.W_enc1[:, is_rare]
rare_mean = rare_enc.mean(-1)
px.histogram(utils.to_numpy(rare_mean @ encoder.W_enc1 / rare_mean.norm() / encoder.W_enc1.norm(dim=0)), title="Cosine Sim with Ave Rare Feature", color=utils.to_numpy(is_rare), labels={"color": "is_rare", "count": "percent", "value": "cosine_sim"}, marginal="box", histnorm="percent", barmode='overlay')

"""## Interpreting A Feature

Let's go and investigate a non rare feature, feature 7
"""

batch_size = 32 # @param {type:"number"}
number_of_batches = 100

"""Let's run the model on some text and then use the autoencoder to process the MLP activations"""

tokens = all_tokens[:batch_size*number_of_batches]
activations = []
layer = 6
site = "resid_post"
act_name = utils.get_act_name(site, layer)

for i in range(number_of_batches):
    with torch.no_grad():
        input_tokens = tokens[i*batch_size:(i+1)*batch_size]
        _, cache = model.run_with_cache(input_tokens, stop_at_layer=layer+1, names_filter=act_name)
        mlp_acts = cache[act_name]
        mlp_acts_flattened = mlp_acts.reshape(-1, cfg["act_size"])

        loss, x_reconstruct, hidden_acts, l2_loss, l1_loss, l0_norm = encoder(mlp_acts_flattened)
        activations.append(hidden_acts.cpu().numpy())
# This is equivalent to:
# hidden_acts = F.relu((mlp_acts_flattened - encoder.b_dec) @ encoder.W_enc + encoder.b_enc)
# print("hidden_acts.shape", hidden_acts.shape)
hidden_acts = np.array(activations).reshape(-1, cfg["dict_size"])

print(hidden_acts.shape)

"""We can now sort and display the top tokens, and we see that this feature activates on text like " and I" (ditto for other connectives and pronouns)! It seems interpretable!

**Aside:** Note on how to read the context column:

A line like "·himself·as·democratic·socialist·and|·he|·favors" means that the preceding 5 tokens are " himself as democratic socialist and", the current token is " he" and the next token is " favors".  · are spaces, ↩ is a newline.

This gets a bit confusing for this feature, since the pipe separators look a lot like a capital I

"""

feature_id = 313
# print(f"Feature freq: {freqs[feature_id].item():.6f}")
token_df = make_token_df(tokens)
features = hidden_acts[:, feature_id]
token_df["feature"] = utils.to_numpy(features)
token_df.sort_values("feature", ascending=False).head(20).style.background_gradient("coolwarm")

"""It's easy to misread evidence like the above, so it's useful to take some text and edit it and see how this changes the model's activations. Here's a hacky interactive tool to play around with some text."""

model.cfg

s = "The 1899 Kentucky gubernatorial election was held on November 7, 1899. The Republican incumbent, William Bradley, was term-limited. The Democrats chose William Goebel. Republicans nominated William Taylor. Taylor won by a vote of 193,714 to 191,331. The vote was challenged on grounds of voter fraud, but the Board of Elections, though stocked with pro-Goebel members, certified the result. Democratic legislators began investigations, but before their committee could report, Goebel was shot by an unknown assassin (event pictured) on January 30, 1900. Democrats voided enough votes to swing the election to Goebel, Taylor was deposed, and Goebel was sworn into office on January 31. He died on February 3. The lieutenant governor of Kentucky, J. C. W. Beckham, became governor, and battled Taylor in court. Beckham won on appeal, and Taylor fled to Indiana, fearing arrest as an accomplice. The only persons convicted in connection with the killing were later pardoned; the assassin's identity remains a mystery"
t = model.to_tokens(s)
print(t)

starting_text = "Hero and I will head to Samantha and Mark's, then he and she will. Then I or you" # @param {type:"string"}
make_feature_vis_gradio(feature_id, starting_text)

"""A final piece of evidence: This is a one layer model, so the neurons can only matter by directly impacting the final logits! We can directly look at how the decoder weights for this feature affect the logits, and see that it boosts `'ll`! This checks out, I and he'll etc is a common construction."""

logit_effect = encoder.W_dec[feature_id] @ model.W_U
create_vocab_df(logit_effect).head(20).style.background_gradient("coolwarm")


sentence = "Enter your sentence here" # @param {type:"string"}
tokens = model.to_tokens(sentence)
_, cache = model.run_with_cache(tokens, stop_at_layer=layer+1, names_filter=act_name)
mlp_acts = cache[act_name]
mlp_acts_flattened = mlp_acts.reshape(-1, cfg["act_size"])
loss, x_reconstruct, hidden_acts, l2_loss, l1_loss, l0_norm = encoder(mlp_acts_flattened)
activations.append(hidden_acts.cpu().numpy())
hidden_acts = np.array(activations).reshape(-1, cfg["dict_size"])
sentence = "I went to the grocery store to buy some"
tokens = model.to_tokens(sentence)

for layer in range(12):
    site = "resid_post"
    act_name = utils.get_act_name(site, layer)


    _, cache = model.run_with_cache(tokens)
    res_acts = cache[act_name]
    res_acts_flattened = res_acts.reshape(-1, cfg["act_size"])
    loss, x_reconstruct, hidden_acts, l2_loss, l1_loss, l0_norm = encoder(res_acts_flattened)
    print(hidden_acts.shape)
    hidden_acts = hidden_acts.detach().cpu().numpy()[-1]

    fig = px.line(hidden_acts)

fig.show()

