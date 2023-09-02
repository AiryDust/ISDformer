import math
import pandas as pd
import numpy as np
import datetime


data = 10*np.sin(np.arange(0, 6545, 0.06545)) + np.arange(0, 6545, 0.06545)
#data = np.repeat([1], 100001)
start = pd.to_datetime('2020-01-01 00:00')
derta = datetime.timedelta(minutes=15)
end = start + derta*100000
date = pd.date_range(start=start, end=end, freq='15min')
out = pd.DataFrame(data, date)
out.to_csv('../data/raise_sine.csv')
