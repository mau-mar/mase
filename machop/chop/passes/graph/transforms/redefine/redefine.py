import torch
import torch.nn as nn
import numpy as np
import copy

from ...utils import get_parent_name


def instantiate_linear(in_features, out_features, bias):
    if bias is not None:
        bias = True
    return nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias)
      
def instantiate_batchnorm(num_features, momentum=0.9):
    return nn.BatchNorm2d(
        num_features=num_features,
        momentum=momentum
    )

def redefine_linear_transform_pass(graph, pass_args=None):
    main_config = copy.deepcopy(pass_args) #pass_args.pop('config')
    default = main_config.pop('default', None)
    if default is None:
        raise ValueError(f"default value must be provided.")
    i = 0
    previous_out = -1
    for node in graph.fx_graph.nodes:
        i += 1
        # if node name is not matched, it won't be tracked
        config = main_config.get(node.name, default)['config']
        name = config.get("name", None)

        if name is not None:
            ori_module = graph.modules[node.target]
            in_features = ori_module.in_features
            out_features = ori_module.out_features
            bias = ori_module.bias
            if name == "output_only":
                out_features = out_features * config["channel_multiplier_out"]
                print(f"Redefining output channel...")
            elif name == "both":
                in_features = previous_out # in_features * config["channel_multiplier_in"]
                out_features = out_features * config["channel_multiplier_out"]
                print(f"Redefining input and output channels...")
            elif name == "input_only":
                in_features = previous_out # in_features * config["channel_multiplier"]
            
            previous_out = out_features
            
            new_module = instantiate_linear(in_features, out_features, bias)
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
        else:
            if node.target in graph.modules:
              ori_module = graph.modules[node.target]

              if isinstance(ori_module, nn.BatchNorm1d):
                if previous_out == -1:
                    previous_out = ori_module.num_features
                else:
                    new_num_features = previous_out
                    new_module = instantiate_batchnorm(new_num_features, ori_module.momentum)
                    previous_out = ori_module.num_features
                    print(ori_module.__dict__)
                    parent_name, name_ = get_parent_name(ori_module.target)
                    setattr(graph.modules[parent_name], name_, new_module)
                  
              elif isinstance(ori_module, nn.Linear):
                if previous_out == -1:
                  previous_out = ori_module.out_features
                else:
                  in_features = previous_out
                  out_features = ori_module.out_features
                  bias = ori_module.bias

                  previous_out = out_features
                  new_module = instantiate_linear(in_features, out_features, bias)
                  parent_name, name = get_parent_name(node.target)
                  setattr(graph.modules[parent_name], name, new_module)
            else:
              continue
 
    return graph, {}

