import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# 1. Load and filter the data
# -----------------------------
# Read ratings CSV (already in 'clean' folder)
ratings = pd.read_csv('clean/ratings.csv')

# Remove users who have rated fewer than 20 books
ratings_filtered = ratings.groupby('user_id').filter(lambda x: len(x) >= 20)

# -----------------------------
# 2. Train/Test Split
# -----------------------------
train_list = []
test_list = []

# For each user, shuffle their ratings and split 20% to test and 80% to train
for user_id, user_ratings in ratings_filtered.groupby('user_id'):
    user_ratings = user_ratings.sample(frac=1, random_state=42).reset_index(drop=True)
    n_test = int(np.ceil(len(user_ratings) * 0.2))
    
    test_ratings = user_ratings.iloc[:n_test]
    train_ratings = user_ratings.iloc[n_test:]
    
    test_list.append(test_ratings)
    train_list.append(train_ratings)

train_df = pd.concat(train_list).reset_index(drop=True)
test_df = pd.concat(test_list).reset_index(drop=True)

# Ensure 'clean' directory exists (optional, if you're certain it already exists you can skip this)
os.makedirs('clean', exist_ok=True)

# Write train and test splits into the 'clean' folder
train_df.to_csv('clean/ratings_train.csv', index=False)
test_df.to_csv('clean/ratings_test.csv', index=False)

# -----------------------------
# 3. Plot Overlapping Histogram
# -----------------------------
# Count how many books each user rated in train and test
train_counts = train_df.groupby('user_id')['book_id'].count()
test_counts = test_df.groupby('user_id')['book_id'].count()

# Determine bin range based on the maximum count
max_count = max(train_counts.max(), test_counts.max())
bins = range(1, max_count + 2)

plt.figure(figsize=(10, 6))

# Plot train histogram
plt.hist(train_counts, bins=bins, alpha=0.5, label='Train', color='blue')

# Plot test histogram
plt.hist(test_counts, bins=bins, alpha=0.5, label='Test', color='red')

# Add labels, title, legend
plt.title('Histogram of Number of Books Rated per User (Train vs. Test)')
plt.xlabel('Number of Books Rated')
plt.ylabel('Number of Users')
plt.legend()

# Save the figure to the 'clean' folder
plt.savefig('clean/train_test_hist.png', dpi=300)

# Display the plot
plt.show()
