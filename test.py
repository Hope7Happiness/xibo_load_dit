import torch
from models import DiT_models
model = DiT_models['DiT-B/4'](input_size=32, num_classes=10)
state_dict = model.state_dict()
print({k: v.shape for k, v in state_dict.items()})

# import 