# clean_tags.py
import os
import pandas as pd
from vllm import LLM
from vllm.sampling_params import SamplingParams

RAW_DIR = "raw"
INSIGHTS_DIR = "insights"
os.makedirs(INSIGHTS_DIR, exist_ok=True)

def classify_tag_gemma(tag: str, llm: LLM) -> str:
    """
    Uses the Gemma 3 model to classify a tag as 1 (useful) or 0 (not useful).

    "1" => Tag is "useful" if it describes a book's genre or sub-genre (romance, sci-fi, etc.)
    "0" => Tag is "not useful" if it does NOT describe a genre, is too specific (e.g. author/title references),
           or if it's in a non-English language.

    Returns a single character, '1' or '0', as a string.
    """
    prompt = f"""
You are an AI assistant helping with a book recommendation system.
We want to classify book tags as either:
"1" = useful (describes a book's genre or sub-genre like romance, sci-fi, fantasy, mystery, etc.)
"0" = not useful (general-purpose, unrelated, overly personal, or overly specific tags)

Strictly return a single character: either 0 or 1.

Rules:
- Classify tags with **non-English characters** as 0.
- Classify **overly specific tags** (e.g. ones that reference authors, book titles, or full series) as 0.
- Classify **personal or meta tags** (like "to-read", "favorite", "read-multiple-times") as 0.

Examples:

Tag: "romance" -> 1
Tag: "romance-bdsm" -> 1
Tag: "mystery" -> 1
Tag: "sci-fi" -> 1
Tag: "fantasy" -> 1
Tag: "distopía" -> 1
Tag: "thriller" -> 1
Tag: "teaching" -> 1

Tag: "favourite" -> 0
Tag: "to-read" -> 0
Tag: "read-multiple-times" -> 0
Tag: "--available-at-raspberrys--" -> 0
Tag: "cb-bjb" -> 0
Tag: "Öykü" -> 0
Tag: "まんが-ほか" -> 0
Tag: "💖" -> 0
Tag: "a-shade-of-vampire-series" -> 0
Tag: "abduction-lisa-gardner-mystery-susp" -> 0
Tag: "books-i-own" -> 0
Tag: "trilogy-book-1" -> 0
Tag: "dystopia-ya-series" -> 0

Now, classify the following tag:

Tag: "{tag}" -> 
""".strip()

    params = SamplingParams(
        temperature=0.2,
        max_tokens=2
    )
    outputs = llm.generate(prompt, params)
    response = outputs[0].outputs[0].text.strip()

    return "1" if "1" in response else "0"


def main():
    # 1) Load raw tags
    tags = pd.read_csv(os.path.join(RAW_DIR, "tags.csv"))
    print("Loaded tags.csv with shape:", tags.shape)

    # 2) Initialize vLLM with the Gemma model
    llm = LLM(
        model="google/gemma-3-27b-it",
        generation_config="vllm",
        tensor_parallel_size=4,
        gpu_memory_utilization=0.90,
        max_model_len=8192,
    )
    print("Gemma 3 model loaded.")

    # 3) Classify each tag, store '0' or '1'
    classifications = []
    for idx, row in tags.iterrows():
        tag_name = row["tag_name"]
        classification = classify_tag_gemma(tag_name, llm)
        classifications.append(classification)

        # Optional: progress
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx+1} / {len(tags)} tags...")

    tags["classification"] = classifications

    # 4) Split tags based on classification
    tags_useful = tags[tags["classification"] == "1"].copy()
    tags_not_useful = tags[tags["classification"] == "0"].copy()

    # 5) Save results
    tags_useful_path = os.path.join(INSIGHTS_DIR, "tags_useful.csv")
    tags_not_useful_path = os.path.join(INSIGHTS_DIR, "tags_not_useful.csv")
    tags_useful.to_csv(tags_useful_path, index=False)
    tags_not_useful.to_csv(tags_not_useful_path, index=False)

    print(f"\nTag classification finished!")
    print(f"  => {len(tags_useful)} rows in tags_useful.csv")
    print(f"  => {len(tags_not_useful)} rows in tags_not_useful.csv")

if __name__ == "__main__":
    main()
