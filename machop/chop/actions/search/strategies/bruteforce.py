import torch
import pandas as pd
import logging
from tabulate import tabulate
import joblib
import copy

from functools import partial

from .base import SearchStrategyBase

#from ....passes.graph.transforms.redefine.redefine import redefine_linear_transform_pass
#from ....passes.graph.analysis.report import init_metadata_analysis_pass
from ....ir import MaseGraph

from ....models import get_model_info, get_model
from ....dataset import MaseDataModule, get_dataset_info


logger = logging.getLogger(__name__)

class SearchStrategyBruteforce(SearchStrategyBase):
    is_iterative = False

    @staticmethod
    def _save_study(study, save_path):
        """
        Save the study object. The subclass can call this method to save the study object at the end of the search.
        """
        with open(save_path, "wb") as f:
            joblib.dump(study, f)

    def _post_init_setup(self):
        self.sum_scaled_metrics = self.config["setup"]["sum_scaled_metrics"]
        self.metric_names = list(sorted(self.config["metrics"].keys()))
        if not self.sum_scaled_metrics:
            self.directions = [
                self.config["metrics"][k]["direction"] for k in self.metric_names
            ]
        else:
            self.direction = self.config["setup"]["direction"]

    def _setup_model(self):
        model = get_model(self.model_name,
                        task="cls",
                        dataset_info=self.dataset_info,
                        pretrained=False)
            
        return model

    def _setup_data(self):
        self.data_module.prepare_data()
        self.data_module.setup()    

    def compute_software_metrics(self, model, sampled_config: dict, is_eval_mode: bool):
        metrics = {}
        if is_eval_mode:
            with torch.no_grad():
                for runner in self.sw_runner:
                    metrics |= runner(self.data_module, model, sampled_config)
        else:
            for runner in self.sw_runner:
                metrics |= runner(self.data_module, model, sampled_config)
        return metrics
    
    '''
    def compute_hardware_metrics(self, model, sampled_config, is_eval_mode: bool):
        metrics = {}
        if is_eval_mode:
            with torch.no_grad():
                for runner in self.hw_runner:
                    metrics |= runner(self.data_module, model, sampled_config)
        else:
            for runner in self.hw_runner:
                metrics |= runner(self.data_module, model, sampled_config)
        return metrics
    '''
    
    def objective(self, search_space, sampled_config, model, is_eval_mode: bool):
        """
        Compute & Scale metrics.
        """
        raise NotImplementedError()

    def search(self, search_space):   
        import itertools     
        results = {}
        best_results = {}

        self._setup_data()

        k = search_space.choices_flattened.keys()

        flattened_lengths = search_space.choice_lengths_flattened.values()

        idx = [list(range(i)) for i in list(flattened_lengths)] #define indexes
    
        configs_list = []
        for combination in itertools.product(*idx):
            combination_config = dict(zip(k, combination))
            configs_list.append(combination_config)

        best_acc = 0
        trial_num = 0
        for i in configs_list:
            sampled_config = search_space.flattened_indexes_to_config(i)
        
            is_eval_mode = self.config.get("eval_mode", True)
            model = search_space.rebuild_model(sampled_config, is_eval_mode)
       
            software_metrics = self.compute_software_metrics(
                model, sampled_config, is_eval_mode
            )
                
            results[f'trial_{trial_num}'] = {'config':sampled_config,
                                        'metrics':software_metrics}
            
            acc = software_metrics['accuracy'] 
            if acc > best_acc:
                best_acc = acc
                best_config = sampled_config
                best_software_metrics = software_metrics
                best_trial = trial_num
                    
                print(f"Best trial Updated: Trial {best_trial} - Best Accuracy: {best_acc}")

            # self.visualizer.log_metrics(metrics=software_metrics, step=i)
            trial_num += 1

        best_results = {
            'config': best_config,
            'metrics':best_software_metrics,
            'acc':best_acc,
            'best_trial':best_trial
        }

        import json
        with open(f'{self.save_dir}/results.json', 'w') as fp:
            json.dump(results, fp)
        with open(f'{self.save_dir}/best_results.json', 'w') as fp:
            json.dump(best_results, fp)

        #self._save_study(results, self.save_dir / "results.json")
        #self._save_study(best_results, self.save_dir / "best_results.json")
    
        print(f"Bruteforce search performed. Saving results in {self.save_dir}/study.json")


        return results 
