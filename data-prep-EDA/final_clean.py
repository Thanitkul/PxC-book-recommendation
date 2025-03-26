import os
import pandas as pd

def main():
    # Ensure the output folder exists
    os.makedirs("clean", exist_ok=True)

    ### 1. Process raw/books.csv ###
    # Read the raw book data.
    book_df = pd.read_csv("raw/books.csv")
    # Create a mapping from goodreads_book_id to book_id (needed later for tag links)
    mapping_df = book_df[['goodreads_book_id', 'book_id']].copy()
    
    # Select only the desired columns.
    # Remove: goodreads_book_id, best_book_id, work_id, books_count, isbn, isbn13,
    #         original_title, work_ratings_count, work_text_reviews_count.
    # Keep only: book_id, authors, original_publication_year, title, language_code,
    #            average_rating, ratings_count, ratings_1, ratings_2, ratings_3, ratings_4,
    #            ratings_5, image_url, small_image_url.
    book_columns = [
        "book_id", "authors", "original_publication_year", "title",
        "language_code", "average_rating", "ratings_count", "ratings_1",
        "ratings_2", "ratings_3", "ratings_4", "ratings_5",
        "image_url", "small_image_url"
    ]
    book_clean = book_df[book_columns]
    book_clean.to_csv("clean/books.csv", index=False)
    
    ### 2. Process raw/ratings.csv ###
    # For ratings, no cleaning is needed.
    ratings_df = pd.read_csv("raw/ratings.csv")
    ratings_df.to_csv("clean/ratings.csv", index=False)
    
    ### 3. Process raw/to_read.csv ###
    # For to_read, no cleaning is needed.
    to_read_df = pd.read_csv("raw/to_read.csv")
    to_read_df.to_csv("clean/to_read.csv", index=False)
    
    ### 4. Process insights/final_merged_tags.csv (Tag Metadata) ###
    # Read the tag metadata file which originally has columns: tag_name, tag_id.
    tags_df = pd.read_csv("insights/final_merged_tags.csv")
    # Swap the columns so that tag_id comes first.
    tags_clean = tags_df[['tag_id', 'tag_name']]
    tags_clean.to_csv("clean/tags.csv", index=False)
    
    ### 5. Process insights/final_book_tags.csv (Book-Tag Links) ###
    # Note: The instructions refer to this file with the same name as above, but for clarity
    # we assume this file is named 'final_book_tags.csv' in the insights folder.
    links_df = pd.read_csv("insights/final_book_tags.csv")
    # links_df contains: goodreads_book_id, tag_id, count.
    # Map the goodreads_book_id to the corresponding book_id using the mapping from raw/books.csv.
    links_merged = pd.merge(links_df, mapping_df, on="goodreads_book_id", how="left")
    # Drop the goodreads_book_id column and reorder columns to: book_id, tag_id, count.
    links_clean = links_merged[['book_id', 'tag_id', 'count']]
    links_clean.to_csv("clean/book_tags.csv", index=False)

if __name__ == '__main__':
    main()
