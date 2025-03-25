import pandas as pd
import matplotlib.pyplot as plt
import os

# Set paths
RAW_DIR = "raw"
INSIGHTS_DIR = "insights"
os.makedirs(INSIGHTS_DIR, exist_ok=True)

# Load data
tags = pd.read_csv(f"{RAW_DIR}/tags.csv")
book_tags = pd.read_csv(f"{RAW_DIR}/book_tags.csv")

# Merge to get tag names with counts
tag_counts = book_tags.groupby("tag_id")["count"].sum().reset_index()
tag_counts = tag_counts.merge(tags, on="tag_id", how="left")

# Sort by frequency
tag_counts_sorted = tag_counts.sort_values("count", ascending=False)

# Save top 50 tags chart
top_tags = tag_counts_sorted.head(52)
# filter out tag "to-read", "currently-reading", and "favorites"
top_tags = top_tags[top_tags["tag_name"] != "-"]
top_tags = top_tags[top_tags["tag_name"] != "to-read"]
top_tags = top_tags[top_tags["tag_name"] != "currently-reading"]
top_tags = top_tags[top_tags["tag_name"] != "favorites"]


plt.figure(figsize=(12, 8))
plt.barh(top_tags["tag_name"][::-1], top_tags["count"][::-1])
plt.title("Top 50 Most Frequent Tags")
plt.xlabel("Total Count")
plt.tight_layout()
plt.savefig(f"{INSIGHTS_DIR}/top_50_tags.png")
plt.close()

# Save bottom 50 tags chart (least used tags)
bottom_tags = tag_counts_sorted[tag_counts_sorted["tag_name"] != "-"].tail(50)
plt.figure(figsize=(12, 8))
plt.barh(bottom_tags["tag_name"], bottom_tags["count"])
plt.title("Bottom 50 Least Frequent Tags")
plt.xlabel("Total Count")
plt.tight_layout()
plt.savefig(f"{INSIGHTS_DIR}/bottom_50_tags.png")
plt.close()

# Save to CSV for reference
tag_counts_sorted.to_csv(f"{INSIGHTS_DIR}/all_tag_counts.csv", index=False)

print("Tag frequency analysis completed and saved in insights/")

# Load books.csv
books = pd.read_csv(f"{RAW_DIR}/books.csv")

# --- Analyze tags for a specific book ---
target_book_id = 2  # Goodreads book ID (not internal book_id)

# Filter book_tags for the given book
book_tag_counts = book_tags[book_tags["goodreads_book_id"] == target_book_id]

# Merge with tag names
book_tag_counts = book_tag_counts.merge(tags, on="tag_id", how="left")

# Sort by count
book_tag_counts_sorted = book_tag_counts.sort_values("count", ascending=False)

# Get book title
book_title_row = books[books["goodreads_book_id"] == target_book_id]
book_title = book_title_row["title"].values[0] if not book_title_row.empty else "Unknown Title"

# Print top tags for the book
print(f"\nTop tags for '{book_title}' (goodreads_book_id = {target_book_id}):")
print(book_tag_counts_sorted[["tag_name", "count"]].head(10))

# Plot top 10 tags
plt.figure(figsize=(10, 6))
plt.barh(book_tag_counts_sorted["tag_name"].head(10)[::-1], book_tag_counts_sorted["count"].head(10)[::-1])
plt.title(f"Top Tags for \"{book_title}\"")
plt.xlabel("Count")
plt.tight_layout()
plt.savefig(f"{INSIGHTS_DIR}/book_{target_book_id}_top_tags.png")
plt.close()
