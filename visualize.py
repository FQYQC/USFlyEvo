import pandas as pd
import numpy as np
from plotly import graph_objects as go

import utils


class Visualizer:
    def __init__(self, dataset):
        self.data = dataset

    def visualize_airport(self, filters={}, item: str = "Passengers"):
        data = self.data.data
        for key, value in filters.items():
            if isinstance(value, list):
                data = data[data[key].isin(value)]
            else:
                data = data[data[key] == value]
        # sum by airport
        data = (
            data.groupby(["Code", "Latitude", "Longitude"])
            .agg({item: "mean"})
            .reset_index()
        )
        data = data[data[item] > 1e-3]
        if item == "Passengers":
            datavalue = np.maximum(np.emath.logn(1.01, data[item]) * 6 + 2000, 0)
        elif item == "Freight tons":
            datavalue = np.maximum(np.emath.logn(1.01, data[item]) * 6 + 3000, 0)
        elif item == "Mail tons":
            datavalue = np.maximum(np.emath.logn(1.01, data[item]) * 6 + 5000, 0)
        fig = go.Figure()
        fig.add_trace(
            go.Scattergeo(
                locationmode="USA-states",
                lon=data["Longitude"],
                lat=data["Latitude"],
                text=data["Code"],
                marker=dict(
                    size=datavalue,
                    sizemode="area",
                    color="rgba(60, 60, 255, 0.5)",
                    line_color="rgba(0, 0, 0, 0)",
                    line_width=0,
                ),
            )
        )
        fig.update_layout(
            title=f"{item} by airport",
            geo=dict(
                scope="usa",
                projection_type="albers usa",
                showlakes=True,
                lakecolor="rgb(220, 220, 220)",
            ),
        )
        return fig

    def visualize_state(self, filters={}, item: str = "Passengers"):
        data = self.data.data
        for key, value in filters.items():
            if isinstance(value, list):
                data = data[data[key].isin(value)]
            else:
                data = data[data[key] == value]
        # sum by state
        data = data.groupby(["State", "Year"])[item].sum().reset_index()
        data = data.groupby("State")[item].mean().reset_index()

        fig = go.Figure()
        fig.add_trace(
            go.Choropleth(
                locations=[utils.STATE_NAME_TO_CODE[state] for state in data["State"]],
                z=data[item],
                locationmode="USA-states",
                colorscale="Viridis",
            )
        )
        fig.update_layout(
            title=f"{item} by state",
            geo_scope="usa",
        )
        return fig


if __name__ == "__main__":
    from dataset.dataset import DataConfig, Dataset
    from regression import Regression
    from models import LinearModel

    dataconfig = DataConfig(root="raw_datasets", save_path="data.npy", cache_save=True)

    dataset = Dataset(dataconfig)

    visualizer = Visualizer(dataset)
    fig = visualizer.visualize_state(
        filters={"Year": 2000}, item="Mail tons"
    )  # filters={"Code": ["LAX", "JFK", "SJC", "SFO", "SEA"]})
    fig.show()
