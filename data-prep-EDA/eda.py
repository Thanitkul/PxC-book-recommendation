import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Handle command-line argument
if len(sys.argv) != 2:
    print("Usage: python eda.py <book_id>")
    sys.exit(1)

try:
    target_book_id = int(sys.argv[1])
except ValueError:
    print("Error: book_id must be an integer.")
    sys.exit(1)

# Directories for input (cleaned CSVs) and output (insights)
CLEAN_DIR = "clean"
INSIGHTS_DIR = "insights"
os.makedirs(INSIGHTS_DIR, exist_ok=True)

# 1. Load tag metadata and book-tag links
tags = pd.read_csv(f"{CLEAN_DIR}/tags.csv")             # columns: tag_id, tag_name
book_tags = pd.read_csv(f"{CLEAN_DIR}/book_tags.csv")   # columns: book_id, tag_id, count

# 2. Group and sum total 'count' by tag_id
tag_counts = book_tags.groupby("tag_id")["count"].sum().reset_index()

# 3. Merge with tags to get tag names
tag_counts = tag_counts.merge(tags, on="tag_id", how="left")

# 4. Sort tags by total frequency
tag_counts_sorted = tag_counts.sort_values("count", ascending=False)

# 5. Save top 50 tags chart
top_tags = tag_counts_sorted.head(50)
plt.figure(figsize=(12, 8))
plt.barh(top_tags["tag_name"][::-1], top_tags["count"][::-1])
plt.title("Top 50 Most Frequent Tags")
plt.xlabel("Total Count")
plt.tight_layout()
plt.savefig(f"{INSIGHTS_DIR}/top_50_tags.png")
plt.close()

# 6. Save bottom 50 tags chart
bottom_tags = tag_counts_sorted[tag_counts_sorted["tag_name"] != "-"].tail(50)
plt.figure(figsize=(12, 8))
plt.barh(bottom_tags["tag_name"], bottom_tags["count"])
plt.title("Bottom 50 Least Frequent Tags")
plt.xlabel("Total Count")
plt.tight_layout()
plt.savefig(f"{INSIGHTS_DIR}/bottom_50_tags.png")
plt.close()

# 7. Save entire tag frequency table
tag_counts_sorted.to_csv(f"{INSIGHTS_DIR}/all_tag_counts.csv", index=False)

print("Tag frequency analysis completed. Charts and CSV saved in 'insights/'.")

# 8. Load books.csv and analyze target book
books = pd.read_csv(f"{CLEAN_DIR}/books.csv")

# Filter tag links for the specific book
book_tag_counts = book_tags[book_tags["book_id"] == target_book_id]

if book_tag_counts.empty:
    print(f"\nNo tags found for book_id {target_book_id}.")
    sys.exit(0)

# Merge with tag names
book_tag_counts = book_tag_counts.merge(tags, on="tag_id", how="left")

# Sort by count
book_tag_counts_sorted = book_tag_counts.sort_values("count", ascending=False)

# Get the book title
book_title_row = books[books["book_id"] == target_book_id]
book_title = book_title_row["title"].values[0] if not book_title_row.empty else "Unknown Title"

# Print top tags for the book
print(f"\nTop tags for '{book_title}' (book_id = {target_book_id}):")
print(book_tag_counts_sorted[["tag_name", "count"]].head(10))

# Plot top tags
plt.figure(figsize=(12, 8))
top_10_tags = book_tag_counts_sorted.head(10)
plt.barh(top_10_tags["tag_name"][::-1], top_10_tags["count"][::-1])
plt.title(f"Top Tags for \"{book_title}\" (book_id={target_book_id})")
plt.xlabel("Count")
plt.tight_layout()
plt.savefig(f"{INSIGHTS_DIR}/book_{target_book_id}_top_tags.png")
plt.close()
