import pandas as pd
import matplotlib.pyplot as plt

# Load the two CSV files without headers
df_new = pd.read_csv(
    "/home/noe/Simulator/autoverse-linux/Linux/Autoverse/Saved/Environments/YasMarina/Data/yas_tbs_left.csv",
    header=None
)
df_old = pd.read_csv(
    "/home/noe/Simulator/autoverse-linux/Linux/Autoverse/Saved/Environments/YasMarina/Data/geo_left.csv",
    header=None
)

# Check the shape of the data
print("New file shape:", df_new.shape)
print("Old file shape:", df_old.shape)

# Example: plot first two columns (x vs y)
plt.figure(figsize=(10, 6))
plt.plot((df_new[0]/100), (df_new[1]/100), label="New (yas_tbs_left)")
plt.plot(df_old[1]-258, df_old[0]-284, label="Old (geo_left)")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Comparison of New vs Old Data")
plt.legend()
plt.grid(True)
plt.show()

#+2.84+2.58
