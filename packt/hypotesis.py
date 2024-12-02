import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp

# Number of people surveyed
n = 4500
# Null hypotesis
pnull= 0.48
# Alternative hypotesis
phat = 0.65

sm.stats.proportions_ztest(phat * n, n, pnull, alternative='larger')
# Our calulated pvalue is 1,22 * 10-126, so we can reject the null hypotesis

sdata = np.random.randint(200, 250, 89)
sm.stats.ztest(sdata, value = 80, alternative = "larger")
sm.stats.ztest(sdata, value = 80, alternative = "larger")

# T-Test
height = np.array([172, 184, 174, 168, 174, 183, 173, 173, 184, 179, 171, 173, 181,
       183, 172, 178, 170, 182, 181, 172, 175, 170, 168, 178, 170, 181,
       180, 173, 183, 180, 177, 181, 171, 173, 171, 182, 180, 170, 172,
       175, 178, 174, 184, 177, 181, 180, 178, 179, 175, 170, 182, 176,
       183, 179, 177])
height

height_average = np.mean(height)
print("Average height is = {0:.3f}".format(height_average))

tset,pval = ttest_1samp(height, 176)

print("P-value = {}".format(pval))

if pval < 0.05:
  print("We are rejecting the null Hypotheis.")
else:
  print("We are accepting the null Hypothesis.")

