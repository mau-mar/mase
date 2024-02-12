# This is the search space for channel multipliers on linear layers.
from copy import deepcopy
from chop.passes.graph.analysis.add_metadata.add_software_metadata import add_software_metadata_analysis_pass
 
from chop.passes.graph.transforms.redefine import redefine_linear_transform_pass, redefine_conv2d_transform_pass
from ..base import SearchSpaceBase
from .....ir.graph.mase_graph import MaseGraph
from .....passes.graph import (
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
)
from .....passes.graph.utils import get_mase_op, get_mase_type
from ..utils import flatten_dict, unflatten_dict
 
DEFAULT_MULTIPLIER_CONFIG = { 
   "config": {
        "name": None
    },
}
 
class Conv2dChannelMultiplierSpace(SearchSpaceBase):
    """
    Search space for channel multipliers on conv2d layers.
    """
 
    def _post_init_setup(self):
        self.model.to("cpu")  # save this copy of the model to cpu
        self.mg = None
        self._node_info = None
        self.default_config = DEFAULT_MULTIPLIER_CONFIG
 
        assert (
            "by" in self.config["setup"]
        ), "Must specify entry `by` (config['setup']['by] = 'name' or 'type')"
 
    def rebuild_model(self, sampled_config, is_eval_mode: bool = True):
        self.model.to(self.accelerator)
        if is_eval_mode:
            self.model.eval()
        else:
            self.model.train()

        mg = MaseGraph(self.model)
        mg, _ = init_metadata_analysis_pass(mg, None)
        mg, _ = add_common_metadata_analysis_pass(
            mg, {"dummy_in": self.dummy_input, "force_device_meta": False}
        )
        mg, _ = add_software_metadata_analysis_pass(mg, None)
        self.mg = mg
        if sampled_config is not None:
            mg, _ = redefine_conv2d_transform_pass(self.mg, sampled_config)
 
        mg.model.to(self.accelerator)
        return mg

    def build_search_space(self):
        mase_graph = self.rebuild_model(sampled_config=None, is_eval_mode=True) # is_eval_mode True/False?
        node_info = {}
        for node in mase_graph.fx_graph.nodes:
            node_info[node.name] = {
                "mase_type": get_mase_type(node),
                "mase_op": get_mase_op(node),
            }
 
        choices = {}
        seed = self.config["seed"]
 
        match self.config["setup"]["by"]:
            case "name":
                # Iterate through all the linear layers in the graph
                for n_name, n_info in node_info.items():
                    n_op = n_info["mase_op"]
                    if n_op in ["conv2d"] or n_op in ["batch_norm2d"]:
                        if n_name in seed:
                            choices[n_name] = deepcopy(seed[n_name])
                        else:
                            choices[n_name] = deepcopy(seed["default"])
            
            #case "type":
            #    # Not suitable for this specific task; potentially useful for future integrations
            #    for n_name, n_info in node_info.items():
            #        n_op = n_info["mase_op"]
            #        if n_op in ["conv2d"]:
            #            if n_op in seed:
            #                choices[n_name] = deepcopy(seed[n_op])
            #            else:
            #                choices[n_name] = deepcopy(seed["default"])
            case _:
                raise ValueError(
                    f"Unknown channel multiplication by: {self.config['setup']['by']}"
                )
 
        # flatten the choices and choice_lengths
        flatten_dict(choices, flattened=self.choices_flattened)
        self.choice_lengths_flattened = {
            k: len(v) for k, v in self.choices_flattened.items()
        }
 
    def flattened_indexes_to_config(self, indexes: dict[str, int]):
        flattened_config = {}
        for k, v in indexes.items():
            flattened_config[k] = self.choices_flattened[k][v]

        config = unflatten_dict(flattened_config)
        config["default"] = self.default_config
        config["by"] = self.config["setup"]["by"]

        ### Make sure the sampled configuration does not lead to size mismatches
        ### For linear channel multiplier search space; must be adapted if needed
        '''
        check = True
        next_out = -1
        for key in config.keys():
          if key == "default" or key == "by":
            continue
          name = config[key]['config']['name']
          if name != "both":
            require_in = config[key]['config']['channel_multiplier']
            require_out = config[key]['config']['channel_multiplier']
          else:
            require_in = config[key]['config']['channel_multiplier_in'] 
            require_out = config[key]['config']['channel_multiplier_out'] 
        
          if next_out != -1:
            check = require_in == next_out
          
          if not check:          
            print(f"Mismatch detected selected by the Sampler; using previous multiplier...")
            print(config)

            if name != "both":
              config[key]['config']['channel_multiplier'] = next_out
            else:
              config[key]['config']['channel_multiplier_in'] = next_out
          next_out = require_out
        '''
        ###

        return config
