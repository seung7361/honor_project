import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/mise.csv")['mise']
plt.plot(df)

output = []

for i in range(len(df) // 7):
    output.append(df[i*7:(i+1)*7].mean())

output = pd.DataFrame(output, columns=['mise'])
output.to_csv('mise_weekly.csv', index=False)