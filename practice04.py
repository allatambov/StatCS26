import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

df = pd.read_csv("https://raw.githubusercontent.com/allatambov/StatCS26/refs/heads/main/Salaries.csv")
df.head()

df.columns = [c.replace(".", "_") for c in  list(df.columns)]
df["salary"] = df["salary"] / 1000

m01 = ols("salary ~ yrs_service", data = df).fit()
print(m01.summary())
print(sm.stats.anova_lm(m01))

m02 = ols("salary ~ yrs_since_phd", data = df).fit()
print(m02.summary())
print(sm.stats.anova_lm(m02))

# presenting results
# pip install stargazer
from stargazer.stargazer import Stargazer

Stargazer([m01, m02])
