# merge_tags.py
import re
import os
import pandas as pd
from nltk.stem import PorterStemmer

# Directories
RAW_DIR = "raw"
INSIGHTS_DIR = "insights"
os.makedirs(INSIGHTS_DIR, exist_ok=True)

# Initialize stemmer
ps = PorterStemmer()

def get_tag_stem(tag: str) -> str:
    """
    Compute a stem for a tag.
    Remove any non-letter characters, lowercase the tag, and apply the Porter stemmer.
    """
    cleaned = re.sub(r'[^a-zA-Z]', '', tag.lower())
    return ps.stem(cleaned)

def get_canonical_tag_simple(tag: str) -> str:
    """
    Returns a canonical genre tag by extracting the first word from the tag.
    Splits the tag on hyphens or underscores and returns the first token in title case.
    
    Example:
      "adventure-action" -> "Adventure"
      "romance-bdsm" -> "Romance"
    """
    tag = tag.strip()
    tokens = re.split(r'[-_]', tag)
    if tokens and tokens[0]:
        return tokens[0].title()
    return tag.title()

def choose_canonical(tags):
    """
    Given a list of tags (strings) that share the same stem,
    choose a canonical tag for display.
    Here we choose the shortest one (could be replaced by a frequency-based choice).
    The canonical tag is returned in title case.
    """
    return sorted(tags, key=len)[0].title()

def get_combined_canonical(tag: str, stem_to_canonical: dict) -> str:
    """
    Combined method:
      - If the tag contains a hyphen or underscore, use the simple extraction.
      - Otherwise, use the stem-based mapping.
    Then, override the candidate with a hardcoded genre mapping if any pattern matches.
    """
    # Use simple extraction if delimiters exist.
    if '-' in tag or '_' in tag:
        candidate = get_canonical_tag_simple(tag)
    else:
        stem = get_tag_stem(tag)
        candidate = stem_to_canonical.get(stem, tag.title())
    
    # Hardcoded mapping for common book genres.
    # Each tuple is (pattern, canonical genre).
    genre_mapping = [
    ("action", "Action"),
    ("advent", "Adventure"),
    ("africa", "African Literature"),
    ("biograph", "Biography"),
    ("chick", "Chick Lit"),
    ("child", "Children"),
    ("classic", "Classics"),
    ("comedy", "Comedy"),
    ("contemp", "Contemporary"),
    ("crime", "Crime"),
    ("detectiv", "Mystery"),
    ("dystop", "Dystopian"),
    ("drama", "Drama"),
    ("fairy", "Fairy Tales"),
    ("fantas", "Fantasy"),
    ("fiction", "Fiction"),
    ("gay", "LGBTQ"),
    ("graphic", "Graphic Novels"),
    ("horror", "Horror"),
    ("histor", "Historical"),
    ("inspir", "Inspirational"),
    ("lesbian", "LGBTQ"),
    ("literat", "Literary Fiction"),
    ("love", "Romance"),
    ("lgbt", "LGBTQ"),
    ("magic", "Fantasy"),
    ("memoir", "Memoir"),
    ("mystery", "Mystery"),
    ("myth", "Mythology"),
    ("nonfic", "Nonfiction"),
    ("paranorm", "Paranormal"),
    ("philosoph", "Philosophy"),
    ("poet", "Poetry"),
    ("postapoc", "Post-Apocalyptic"),
    ("psycholog", "Psychology"),
    ("relig", "Religion"),
    ("romanc", "Romance"),
    ("sci", "Science Fiction"),
    ("selfhelp", "Self-Help"),
    ("short", "Short Stories"),
    ("spirit", "Spirituality"),
    ("sports", "Sports"),
    ("suspens", "Thriller"),
    ("thrill", "Thriller"),
    ("travel", "Travel"),
    ("urban", "Urban Fiction"),
    ("vamp", "Vampires"),
    ("war", "War"),
    ("witch", "Witches"),
    ("ya", "Young Adult"),
    ("youngadult", "Young Adult"),
    ("zomb", "Zombies"),
]

    
    lower_candidate = candidate.lower()
    for pattern, genre in genre_mapping:
        if pattern in lower_candidate:
            return genre
    return candidate

def main():
    # 1. Load the useful tags file (output from step 1)
    df_tags = pd.read_csv(os.path.join(INSIGHTS_DIR, "tags_useful.csv"))
    # Expecting columns: tag_id, tag_name, classification
    print("Loaded tags_useful.csv with shape:", df_tags.shape)

    # 2. Compute a stem for each tag (for the stemming-based grouping).
    df_tags["stem"] = df_tags["tag_name"].apply(get_tag_stem)
    
    # 3. For each stem, choose a canonical tag using the stemming method.
    stem_to_canonical = (
        df_tags.groupby("stem")["tag_name"]
               .apply(lambda tags: choose_canonical(list(tags)))
               .to_dict()
    )
    
    # 4. Apply the combined method to get the final canonical tag.
    df_tags["canonical_tag"] = df_tags["tag_name"].apply(
        lambda tag: get_combined_canonical(tag, stem_to_canonical)
    )
    
    # Build a mapping dictionary: original tag -> canonical tag.
    mapping_dict = dict(zip(df_tags["tag_name"], df_tags["canonical_tag"]))
    
    # 5. Create the final merged tags dataframe by dropping duplicates based on canonical_tag.
    final_tags = df_tags[["canonical_tag"]].drop_duplicates().reset_index(drop=True)
    # Assign new tag IDs (starting at 1)
    final_tags["new_tag_id"] = final_tags.index + 1
    # Rename columns for output
    final_merged_tags = final_tags.rename(columns={"new_tag_id": "tag_id", "canonical_tag": "tag_name"})
    
    # Save final merged tags CSV.
    final_merged_tags.to_csv(os.path.join(INSIGHTS_DIR, "final_merged_tags.csv"), index=False)
    
    # 6. Process the book tags.
    # Load the original book_tags.csv (expected columns: goodreads_book_id, tag_id, count)
    df_book_tags = pd.read_csv(os.path.join(RAW_DIR, "book_tags.csv"))
    
    # Filter book_tags to include only those tag_ids that are present in our useful tags file.
    useful_tag_ids = set(df_tags["tag_id"])
    df_book_tags_useful = df_book_tags[df_book_tags["tag_id"].isin(useful_tag_ids)].copy()
    
    # Merge in the canonical tag from df_tags (via tag_id).
    df_book_tags_useful = df_book_tags_useful.merge(
        df_tags[["tag_id", "canonical_tag"]],
        on="tag_id",
        how="left"
    )
    
    # Build a dictionary from canonical tag to new tag id.
    canonical_to_new_id = dict(zip(final_merged_tags["tag_name"], final_merged_tags["tag_id"]))
    
    # Map the canonical tag to new tag id in the book tags.
    df_book_tags_useful["new_tag_id"] = df_book_tags_useful["canonical_tag"].map(canonical_to_new_id)
    
    # Group by goodreads_book_id and new_tag_id and sum the counts.
    final_book_tags = df_book_tags_useful.groupby(
        ["goodreads_book_id", "new_tag_id"], as_index=False
    )["count"].sum().rename(columns={"new_tag_id": "tag_id"})
    
    # Save the final book tags CSV.
    final_book_tags.to_csv(os.path.join(INSIGHTS_DIR, "final_book_tags.csv"), index=False)
    
    # 7. Output mappings.
    print("Final merged tags and final book tags saved.")
    print("\nMapping dictionary (original tag -> canonical tag):")
    print(mapping_dict)
    print("\nCanonical to new tag id mapping:")
    print(canonical_to_new_id)

if __name__ == "__main__":
    main()
