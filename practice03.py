import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

flats = pd.read_csv("https://raw.githubusercontent.com/allatambov/StatCS26/refs/heads/main/flats_upd.csv")
model = ols("lprice ~ square", data = flats).fit()
print(model.summary())
print(sm.stats.anova_lm(model))

