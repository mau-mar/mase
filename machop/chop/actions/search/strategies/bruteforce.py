import torch
import pandas as pd
import logging
from tabulate import tabulate
import joblib
import copy

from functools import partial

from .base import SearchStrategyBase

from ....passes.graph.transforms.redefine.redefine import redefine_linear_transform_pass
from ....passes.graph.analysis.report import init_metadata_analysis_pass
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

    def _setup_model_mg(self):
        model = get_model(self.model_name,
                        task="cls",
                        dataset_info=self.dataset_info,
                        pretrained=False)
            
        mg = MaseGraph(model=model)
        mg, _ = init_metadata_analysis_pass(mg, None)
        return model, mg

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

    def search(self, search_space):        
        results = {}

        self._setup_data()

        best_acc = 0

        for i, config in enumerate(search_space):
            sampled_config = copy.deepcopy(config)

            model, mg = self._setup_model_mg()
            # mg, _ = redefine_linear_transform_pass(mg, {"config": sampled_config})
            # model = mg.model

            is_eval_mode = self.config.get("eval_mode", False)

            model = search_space.rebuild_model(sampled_config, is_eval_mode)

            software_metrics = self.compute_software_metrics(
                model, sampled_config, is_eval_mode
            )
            # hardware_metrics = self.compute_hardware_metrics(
            #     model, sampled_config, is_eval_mode
            # )

            self.visualizer.log_metrics(metrics=software_metrics, step=i)

            trial_acc = ...
            if trial_acc > best_acc:
                best_acc = trial_acc
                best_config = sampled_config
                best_multiplier_1 = config['seq_blocks_2']['config']['channel_multiplier']
                best_multiplier_2 = config['seq_blocks_6']['config']['channel_multiplier']

            results[f'trial_{i}'] = {'config':sampled_config,
                                     'metrics':software_metrics}

        best_results = {'config':best_config,
                        'test_accuracy':best_acc}

        print(f"Best Accuracy: {best_acc}\nBest Channel Multiplier 1: {best_multiplier_1}\nBest Channel Multiplier 2: {best_multiplier_2}")

        self._save_study(results, self.save_dir / "study.json")
        self._save_study(best_results, self.save_dir / "best.json")

        return results, best_results #best_acc, best_multiplier_1, best_multiplier_2, recorded_accs
