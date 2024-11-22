from collections import OrderedDict
from typing import List

import torch
import numpy as np


def set_num_batches_in_state_dict(state_dict):
    for key in state_dict.keys():
        try:
            if state_dict[key].numel() == 0:
                state_dict[key] = torch.tensor(0, dtype=torch.long)
        except Exception:
            continue
    return state_dict


# class Utils:
#
#     @staticmethod
#     def get_parameters(net) -> List[np.ndarray]:
#         return [val.cpu().numpy() for _, val in net.state_dict().items()]
#
#     @staticmethod
#     def set_parameters(net, parameters: List[np.ndarray]):
#         params_dict = zip(net.state_dict().keys(), parameters)
#         state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
#
#         state_dict = set_num_batches_in_state_dict(state_dict)
#
#         net.load_state_dict(state_dict, strict=True)
#
#         return net
