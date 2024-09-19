import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["toolbar"] = "None"
# Load your dataset
# For example, assume it's a CSV file with columns 'Year' and 'Population'
df = pd.DataFrame(
    {"Year": np.arange(0, 2000), "Population": np.linspace(0, 1000000, 2000)}
)

# Initialize the figure and axis
fig, ax = plt.subplots()
(line,) = ax.plot([], [], lw=2)
ax.set_xlabel("Year")
ax.set_ylabel("Population")
ax.set_title("US Population Growth Over Time")

def animate(i):
    x = df["Year"][:i]
    y = df["Population"][:i]
    line.set_data(x, y)
    if i > 0:
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
    return (line,)


ani = animation.FuncAnimation(
    fig,
    animate,
    frames=len(df),
    interval=1000,
    repeat=False,
)

plt.show()
