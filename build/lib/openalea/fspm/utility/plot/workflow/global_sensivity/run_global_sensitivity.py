import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  r2_score


def regression_analysis(dataset, output_path, extract_prop):
    # TODO : normalize and comment
    regression_variables = [dim for dim in dataset.dims if dim not in ('t', 'vid')]
    df_regression = pd.DataFrame(columns=['variable', 'r2', 'intercept'] + regression_variables)
    for global_output in extract_prop.keys():
        y = dataset.sel(vid=1)[global_output]
        y = y.sel(t=max(y.t))
        y = y.stack(stk=y.dims)
        y = (y - y.min()) / (y.max() - y.min())
        y = y.fillna(0)
        x = y.coords['stk'].to_numpy()
        x = [list(k) for k in x]
        y = list(y.to_numpy())

        regressor = LinearRegression()
        regressor.fit([list(k) for k in x], y)
        y_pred = regressor.predict(x)
        r2 = r2_score(y, y_pred)

        keys = ['variable', 'r2', 'intercept'] + regression_variables
        values = [global_output, r2, regressor.intercept_] + [coef for coef in regressor.coef_]

        df_regression.loc[len(df_regression)] = dict(zip(keys, values))

    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    table = pd.plotting.table(ax, df_regression, loc='upper right')
    table.auto_set_font_size(True)
    plt.savefig(output_path + '/linear_regression.png', dpi=300)

    return