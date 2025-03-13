import jax
import os
import argparse
from flax.training import checkpoints
import orbax.checkpoint as ocp
import torch
from models import DiT_models

def jax_to_torch(arr, conv=False, linear=False):
    if linear:
        arr = arr.transpose((1, 0))
    if conv:
        arr = arr.transpose((3, 2, 0, 1))
    return (torch.from_numpy(jax.device_get(arr)), 'new')

def load_mlp(torch_obj, key, jax_obj):
    torch_obj[key+'.weight'] = jax_to_torch(jax_obj['_flax_linear']['kernel'], linear=True)
    torch_obj[key+'.bias'] = jax_to_torch(jax_obj['_flax_linear']['bias'])

def load_block(torch_obj, key, jax_obj):
    load_mlp(torch_obj, key+'.adaLN_modulation.1', jax_obj['adaLN_modulation']['layers_1'])
    load_mlp(torch_obj, key+'.attn.qkv', jax_obj['attn']['qkv'])
    load_mlp(torch_obj, key+'.attn.proj', jax_obj['attn']['proj'])
    load_mlp(torch_obj, key+'.mlp.fc1', jax_obj['mlp']['fc1'])
    load_mlp(torch_obj, key+'.mlp.fc2', jax_obj['mlp']['fc2'])

# load jax model
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to pretrained checkpoint folder', required=True)
args = parser.parse_args()
path = args.path
if path.endswith('/'):
    path = path[:-1]
assert os.path.exists(path)
state = checkpoints.restore_checkpoint(os.path.abspath(path), target=None)
params = state['ema_params']['net']
# print(jax.tree_map(lambda x: x.shape, params))

# load torch model
model = DiT_models['DiT-B/4'](input_size=32, num_classes=1000, in_channels=3, learn_sigma=False)
state_dict = model.state_dict()
# print({k: v.shape for k, v in state_dict.items()})
old_shapes = {k: v.shape for k, v in state_dict.items()}
state_dict = {k: (v, 'old') for k, v in state_dict.items()}
state_dict['pos_embed'] = (state_dict['pos_embed'][0], 'new')

# main load part

### load x_embedder
state_dict["x_embedder.proj.weight"] = jax_to_torch(params["x_embedder"]["proj"]["kernel"], conv=True)
state_dict["x_embedder.proj.bias"] = jax_to_torch(params["x_embedder"]["proj"]["bias"])

### load y_embedder
state_dict["y_embedder.embedding_table.weight"] = jax_to_torch(params["y_embedder"]["embedding_table"]["_flax_embedding"]["embedding"])

### load t_embedder
load_mlp(state_dict, "t_embedder.mlp.0", params["t_embedder"]["mlp"]["layers_0"])
load_mlp(state_dict, "t_embedder.mlp.2", params["t_embedder"]["mlp"]["layers_2"])

### load final layer
load_mlp(state_dict, "final_layer.adaLN_modulation.1", params["final_layer"]["adaLN_modulation"]["layers_1"])
load_mlp(state_dict, "final_layer.linear", params["final_layer"]["linear"])

### load transformer
n_layers = len(params["blocks"])
for i in range(n_layers):
    load_block(state_dict, f"blocks.{i}", params["blocks"][f'layers_{i}'])

bad_keys = {k for k, v in state_dict.items() if v[1] == 'old'}
if len(bad_keys) > 0:
    print('bad keys: ', bad_keys)
    exit(1)
    
wrong_keys = {k for k, v in state_dict.items() if v[0].shape != old_shapes[k]}
if len(wrong_keys) > 0:
    print('wrong keys: ', {k: (v[0].shape, old_shapes[k]) for k, v in state_dict.items() if k in wrong_keys})
    exit(2)

state_dict = {k: v[0] for k, v in state_dict.items()}

assert not os.path.isabs(path)
# assert path.startswith('jax_models/')
dirname, basename = os.path.split(path)
save_path = os.path.join('torch_models', basename+'.pt')
print('saving model to ', save_path)
# print(state.keys())
torch.save(state_dict, save_path)
print('Success!')