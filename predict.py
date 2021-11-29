import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

from data_processing import getData


def rolling_window(a, window):
    """from https://stackoverflow.com/questions/27852343/split-python-sequence-time-series-array-into-subsequences-with-overlap"""
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def weight(var1, var2):
    mean_var1, mean_var2 = np.mean(var1), np.mean(var2, axis=1)
    std_var1, std_var2 = np.std(var1), np.std(var2, axis=1)
    # if std_var2 is 0, this will give a NaN, which is why we need to replace it with 0
    correlation = (
        1
        / (len(var1))
        * (
            np.sum((var1 - mean_var1) * (var2 - mean_var2[:, np.newaxis]), axis=1)
            / (std_var2 * std_var1)
        )
    )
    return np.nan_to_num(correlation * (std_var2 / std_var1))


def getTrendForAll(arr, window_size):
    rolling = rolling_window(arr, window_size)[1:]
    w_1 = weight(np.arange(window_size), rolling)
    fill = np.empty((window_size,))
    fill[:] = np.nan
    return np.append(w_1, fill)


def add_attribute(df, column_to_add, k):
    copy = df.copy()
    for i in range(1, k + 1):
        filler = np.empty((i,))
        filler[:] = np.nan
        copy["{}_t-{}".format(column_to_add, i)] = np.append(
            filler, copy[column_to_add].iloc[:-i].to_numpy()
        )

    return copy


class Model:
    def __init__(self, df) -> None:
        self.df = df
        self.window_size = 90
        self.scale = 1e6
        self.k = 30
        train_x, train_y = self.prepareData()

        # train model
        best_penalty = 0.00045203536563602405
        self.reg = Lasso(alpha=best_penalty, normalize=True, max_iter=1e5).fit(
            train_x, train_y
        )

    def prepareData(self):
        # get percentage of missing values for each column
        temp = (self.df.isna().sum() / len(self.df)).to_frame()

        # Regression cannot handle NaN-values, so we'll have to get rid of the columns that have too many of them
        threshhold = 0.1
        cols_to_keep = temp[(temp < threshhold).to_numpy()].index
        self.df = self.df[cols_to_keep]
        # we'll replace the remaining missing values with the mean of that column
        self.df.fillna(self.df.mean(), inplace=True)

        trends = list(
            self.df.groupby("Location").apply(
                lambda x: getTrendForAll(
                    (x["New_cases"] * self.scale / x["Population"]).to_numpy(),
                    self.window_size,
                )
            )
        )
        trends = [elem for sublist in trends for elem in sublist]
        self.df["trend"] = trends

        non_numerical_cols = ["Location", "Date", "Weekday", "Continent", "CountryCode"]
        self.diff_cols = self.df.columns.difference(["trend"] + non_numerical_cols)

        self.df = add_attribute(self.df, "New_cases", self.k)

        split = 0.6, 0.2, 0.2
        train_data = self.df.groupby("Location").apply(
            lambda x: x[
                (x["Date"] > x["Date"].min() + pd.Timedelta(self.k, "days"))
                & (
                    x["Date"]
                    < x["Date"].max()
                    - pd.Timedelta(self.window_size, "days")
                    - (
                        (
                            x["Date"].max()
                            - pd.Timedelta(self.window_size, "days")
                            - (x["Date"].min() + pd.Timedelta(self.k, "days"))
                        )
                        * (split[1] + split[2])
                    )
                )
            ]
        )

        train_data_X, train_data_y = train_data[self.diff_cols], train_data["trend"]

        return train_data_X, train_data_y

    def predict(self, country, num_last_days=90):
        temp = self.df[self.df["Location"] == country]
        pred = self.reg.predict(temp.iloc[-2:-1][self.diff_cols])
        pred_x = [
            temp.iloc[-1]["Date"] + pd.Timedelta(i, "days")
            for i in range(self.window_size + 1)
        ]
        # convert prediction into actual values
        pred_y = temp.iloc[-1]["New_cases"] + np.arange(self.window_size + 1) * pred
        hist = temp[
            (
                temp["Date"]
                >= temp.iloc[-1]["Date"] - pd.Timedelta(num_last_days, "days")
            )
        ]

        return pred_x, pred_y, hist["Date"], hist["New_cases"]


if __name__ == "__main__":
    df = getData()

    model = Model(df)
    pred_x, pred_y, hist_x, hist_y = model.predict("Norway")
    plt.plot(pred_x, pred_y, label="prediction")
    plt.plot(hist_x, hist_y, label="history")
    plt.show()
