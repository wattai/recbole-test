import os

import pandas as pd
from src.dataset.base_dataset import BaseDataset  # https://github.com/RUCAIBox/RecSysDatasets/blob/master/conversion_tools/src/base_dataset.py をコピーして所定の場所に配置しておく

class CookpadMartDataset(BaseDataset):
    def __init__(self, input_path, output_path):
        super(CookpadMartDataset, self).__init__(input_path, output_path)
        self.dataset_name = "ckpd_mart"

        # input_path
        self.interact_file = os.path.join(self.input_path, "interact.csv")
        self.item_file = os.path.join(self.input_path, "items.csv")
        self.user_file = os.path.join(self.input_path, "users.csv")

        self.sep = ","

        # output_path
        output_files = self.get_output_files()
        self.output_interact_file = output_files[0]
        self.output_item_file = output_files[1]
        self.output_user_file = output_files[2]

        # selected feature fields
        # 型について -> https://recbole.io/docs/user_guide/data/atomic_files.html#format
        self.interact_fields = {
            0: "user_id:token",
            1: "item_id:token",
            2: "timestamp:float",
        }

        self.item_fields = {
            0: "item_id:token",
            1: "item_name:token",
            2: "item_category_id:token"
        }

        self.user_fields = {
            0: "user_id:token",
            1: "feature1:token",
            2: "feature2:token",
        }

    def load_inter_data(self):
        return pd.read_csv(self.interact_file, delimiter=self.sep, engine="python")

    def load_item_data(self):
        return pd.read_csv(self.item_file, delimiter=self.sep, engine="python")

    def load_user_data(self):
        return pd.read_csv(self.user_file, delimiter=self.sep, engine="python")

