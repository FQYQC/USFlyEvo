import pandas as pd
import os
import numpy as np
import os.path as osp


class DataConfig:
    start_year = 1990
    end_year = 2022
    num_years = end_year - start_year + 1
    unit = 1000000
    base_year = 1990
    exclude_non_us50 = True

    def __init__(self, root, cache_path=None, cache_save=False, save_path=None):
        self.root = root
        self.cache_path = osp.join(root, cache_path) if cache_path is not None else None
        self.cache_save = cache_save
        self.save_path = osp.join(root, save_path) if save_path is not None else None


class Dataset:
    def __init__(self, config: DataConfig):
        self.dataconfig = config
        self._data = None
        self._non_us50_area = [
            "U.S. Pacific Trust Territories",
            "U.S. Pacific Trust Territories and Possessions",
            "U.S. Virgin Islands",
            "Puerto Rico",
        ]

        if config.cache_path is not None and osp.exists(config.cache_path):
            self._data = np.load(config.cache_path, allow_pickle=True).item()
        else:
            self._data = self._load_data()
            if config.cache_save:
                np.save(config.save_path, self._data)

    def _load_data(self):
        data = []
        states_list = []
        is_state = []
        for i in range(1, 55):
            path = osp.join(self.dataconfig.root, f"_table_Full Data_data ({i}).csv")
            if not osp.exists(path):
                print(f"File {path} not found")
                continue
            da = pd.read_csv(path)
            if self.dataconfig.exclude_non_us50 and da["State"][0] in self._non_us50_area:
                continue
            data.append(da)
            states_list.append(da["State"][0])
            is_state.append(False if da["State"][0] in self._non_us50_area else True)

        data = pd.concat(data)
        data["Passengers"] /= self.dataconfig.unit
        data["Freight tons"] /= self.dataconfig.unit
        data["Mail tons"] /= self.dataconfig.unit

        return {
            "data": data,
            "states_list": states_list,
            "is_state": is_state,
        }

    @property
    def data(self):
        return self._data["data"]

    @property
    def states_list(self):
        return self._data["states_list"]

    # Select functions
    # State,City,Airport,Code,Year,Freight tons,Mail tons,Passengers,Latitude,Longitude

    def select_airport(self, airport_code, table=None):
        if table is not None:
            return table[table["Code"] == airport_code]
        return self.data[self.data["Code"] == airport_code]

    def select_state(self, state_name, table=None):
        if table is not None:
            return table[table["State"] == state_name]
        return self.data[self.data["State"] == state_name]

    def select_city(self, city_name, table=None):
        if table is not None:
            return table[table["City"] == city_name]
        return self.data[self.data["City"] == city_name]

    def select_year(self, year, table=None):
        if table is not None:
            return table[table["Year"] == year]
        return self.data[self.data["Year"] == year]

    def select_region(self, lat, lon, lat_range, lon_range, table=None):
        if table is not None:
            return table[
                (table["Latitude"] >= lat - lat_range)
                & (table["Latitude"] <= lat + lat_range)
                & (table["Longitude"] >= lon - lon_range)
                & (table["Longitude"] <= lon + lon_range)
            ]
        return self.data[
            (self.data["Latitude"] >= lat - lat_range)
            & (self.data["Latitude"] <= lat + lat_range)
            & (self.data["Longitude"] >= lon - lon_range)
            & (self.data["Longitude"] <= lon + lon_range)
        ]


if __name__ == "__main__":
    dataset = Dataset("raw_datasets")
    print(dataset._data["states_list"])
