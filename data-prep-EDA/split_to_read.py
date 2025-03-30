import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Read the Wishlist Data ---
df = pd.read_csv('clean/to_read.csv')

# --- Step 2: Identify Eligible Users ---
# Count wishlist entries per user
user_counts = df.groupby('user_id').size()

# Eligible users: those with more than 5 wishlist entries
eligible_users = user_counts[user_counts > 5].index

# Randomly sample 20% of the eligible users for the test set
np.random.seed(42)  # for reproducibility
test_sample_size = int(len(eligible_users) * 0.2)
test_users = np.random.choice(eligible_users, size=test_sample_size, replace=False)

# Train users: All users not in test_users (this includes ineligible users)
train_users = df['user_id'].unique()[~np.isin(df['user_id'].unique(), test_users)]

# --- Step 3: Create Train and Test DataFrames ---
train_df = df[df['user_id'].isin(train_users)]
test_df = df[df['user_id'].isin(test_users)]

# Save the splits to CSV files
train_df.to_csv('clean/to_read_train.csv', index=False)
test_df.to_csv('clean/to_read_test.csv', index=False)

# --- Step 4: Compute and Print Summary Statistics ---
# Count wishlist entries per user in each split
train_counts = train_df.groupby('user_id').size()
test_counts = test_df.groupby('user_id').size()

print("Train Wishlist Summary:")
print(f"Total wishlist entries: {len(train_df)}")
print(f"Unique users: {train_df['user_id'].nunique()}")
print(f"Average wishlist entries per user: {train_counts.mean():.2f}")

print("\nTest Wishlist Summary:")
print(f"Total wishlist entries: {len(test_df)}")
print(f"Unique users: {test_df['user_id'].nunique()}")
print(f"Average wishlist entries per user: {test_counts.mean():.2f}")

# --- Step 5: Plot Histogram of Wishlist Counts per User for Both Splits ---
plt.figure(figsize=(10, 6))
# Determine bins based on the maximum wishlist count from both splits
bins = range(1, max(train_counts.max(), test_counts.max()) + 2)

plt.hist(train_counts, bins=bins, alpha=0.7, label='Train')
plt.hist(test_counts, bins=bins, alpha=0.7, label='Test')
plt.xlabel('Number of Wishlist Entries per User')
plt.ylabel('Number of Users')
plt.title('Distribution of Wishlist Entries per User (Train vs Test)')
plt.legend()
plt.xticks(bins)
plt.show()

# Save the histogram to the 'clean' folder
plt.savefig('clean/wishlist_distribution.png')
