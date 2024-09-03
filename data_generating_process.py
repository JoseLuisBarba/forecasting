import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import timesynth as ts

def fx_plot_time_serie(
        time: np.ndarray, values: np.ndarray, title: str
    ) -> None:
    fig = go.Figure(
        data= go.Scatter(x=time, y=values, mode="lines", name=title)
    )
    fig.update_layout(
        title=title, xaxis_title= "Time", yaxis_title= "Values"
    )
    fig.show()

# white and red noise:
def fx_white_noise():
    time = np.arange(200)
    # sample
    # mean 0, std_dev = 100
    values = 0 + np.random.randn(200) * 100
    title = "White noise process"
    fx_plot_time_serie(time, values, title)

def fx_red_noise():
    # correlation coeffient
    r = 0.4
    time  = np.arange(stop=200)
    white_noise = 0 + np.random.rand(200) * 100
    # create red noise by intriducing correlation
    # between subsequent values in the white noise
    values = np.zeros(200)
    for i, v in enumerate(white_noise):
        if i == 0:
            values[i] = v 
        else:
            values[i] = r*values[i-1] + np.sqrt( 1 - np.power(r,2))*v
    title = "Red Noise Process"
    fx_plot_time_serie(time, values, title)


def generate_timeseries(signal, noise=None):
    time_sampler = ts.TimeSampler(stop_time=20)
    regular_time_samples = time_sampler.sample_regular_time(num_points=100)
    timeseries = ts.TimeSeries(signal_generator=signal, noise_generator=noise)
    samples, signals, errors = timeseries.sample(regular_time_samples)
    return samples, regular_time_samples, signals, errors


