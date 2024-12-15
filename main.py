import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

import utils
from dataset.dataset import DataConfig, Dataset
from regression import Regression
from visualize import Visualizer
from models import *

if os.path.exists("raw_datasets/data.npy"):
    dataconfig = DataConfig(root="raw_datasets", cache_path="data.npy")
else:
    dataconfig = DataConfig(root="raw_datasets", save_path="data.npy", cache_save=True)
dataset = Dataset(dataconfig)

logger = utils.Logger("output.txt", overwrite=True, echo_to_stdout=True)
visualizer = Visualizer(dataset)
parameters = {}

# Section 4.1
representative_airports = ["ATL", "DEN", "DFW", "JFK", "LAX", "ORD", "SEA", "ALL CA"]

for item in ["Passengers", "Freight tons", "Mail tons"]:
    for airport in representative_airports:
        if airport == "ALL CA":
            airport_data = dataset.select_state("California")
            airport_data = airport_data.groupby("Year").sum().reset_index()
        else:
            airport_data = dataset.select_airport(airport)

        x, y = airport_data["Year"], airport_data[item]
        x, y = x.to_numpy(), y.to_numpy()
        x = x[:, np.newaxis]
        x = x - DataConfig.start_year

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x.squeeze() + dataconfig.start_year, y=y, mode="markers", name="Data"
            )
        )

        model = LinearGaussianModel([0, 0, 2], None)

        regression = Regression(model, model.objective)
        regression.fit(x[:-3], y[:-3])
        if item == "Passengers":
            parameters[airport] = regression.model.params
        logger.log(
            f"Airport: {airport}, w/o covid, items: {item}, params: {model.params}"
        )
        fig.add_trace(
            go.Scatter(
                x=x.squeeze() + dataconfig.start_year,
                y=model(x),
                mode="lines",
                name="Prediction w/o 2020+",
                line=dict(color="red"),
            )
        )

        regression.fit(x, y)
        logger.log(
            f"Airport: {airport}, w covid, items: {item}, params: {model.params}"
        )
        fig.add_trace(
            go.Scatter(
                x=x.squeeze() + dataconfig.start_year,
                y=model(x),
                mode="lines",
                name="Prediction w 2020+",
                line=dict(color="blue"),
            )
        )

        fig.update_layout(
            # title=f"Passengers by year for {airport}",
            # title_font_size=40,
            xaxis_title="Year",
            yaxis_title=f"{item}/million",
            yaxis=dict(
                title_font=dict(size=25),
                range=[0, 120] if airport == "ALL CA" else [0, 60],
            ),
            xaxis=dict(title_font=dict(size=25)),
            title_x=0.5,
            title_y=0.9,
            legend=dict(
                yanchor="top", y=0.99, xanchor="left", x=0.01, title_font_size=25
            ),
        )
        visualizer.save_fig(fig, f"plots/{airport}_{item}.png")
        # fig.show()

# Section 4.2
for airport in representative_airports:
    if airport == "ALL CA":
        airport_data = dataset.select_state("California")
        airport_data = airport_data.groupby("Year").sum().reset_index()
    else:
        airport_data = dataset.select_airport(airport)

    model.params = parameters[airport]
    
    x, y = airport_data["Year"], airport_data["Passengers"]
    x, y = x.to_numpy(), y.to_numpy()
    x = x[:, np.newaxis]
    x = x - DataConfig.start_year

    x = x[-3:]
    y = y[-3:]
    
    y_hat = model(x)
    print(y_hat, y)
    z = (y - 0.7 * y_hat).sum() / np.sqrt(3) / model.params[-1]
    p = (norm.cdf(z))
    np.set_printoptions(precision=3)
    logger.log(f"Airport: {airport}, p-value: {p}")
