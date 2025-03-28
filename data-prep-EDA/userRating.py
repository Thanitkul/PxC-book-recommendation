import pandas as pd
import matplotlib.pyplot as plt

# Load the ratings.csv file
ratings = pd.read_csv("./clean/ratings.csv")

# Count how many books each user rated
user_rating_counts = ratings['user_id'].value_counts()

print(user_rating_counts)
# Plot the histogram
plt.figure(figsize=(10, 6))
plt.hist(user_rating_counts, bins=range(1, user_rating_counts.max() + 2), edgecolor='black', align='left')
plt.title("Histogram of Number of Books Rated per User")
plt.xlabel("Number of Books Rated")
plt.ylabel("Number of Users")

# Set x-axis ticks with a specific interval (e.g., every 5)
interval = 20
plt.xticks(range(1, user_rating_counts.max() + 1, interval))

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save plot to insights folder
plt.savefig("insights/user_rating_histogram.png")
