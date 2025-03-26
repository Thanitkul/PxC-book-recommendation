# clean_tags.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
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
    "0" => Tag is "not useful" if it does NOT describe a genre (favorites, to-read, random, etc.).

    Returns a single character, '1' or '0', as a string.
    """
    prompt = f"""
You are an AI assistant helping with a book recommendation system. 
We have a tag that people use to describe books, and we want to classify it as either
"1" = useful or "0" = not useful, strictly returning a single digit "0" or "1".

We define "useful" if it describes a book's genre or sub-genre (like romance, sci-fi, fantasy, etc.). 
"Not useful" are tags that do not describe a genre (like to-read, favourites, random personal labels, etc.).

Here are some examples:

Tag: "romance" -> 1
Tag: "romance-bdsm" -> 1
Tag: "teaching" -> 1
Tag: "distopía" -> 1
Tag: "distributed" -> 0
Tag: "favourite" -> 0
Tag: "--available-at-raspberrys--" -> 0
Tag: "cb-bjb" -> 0
Tag: "read-multiple-times" -> 0

Now, classify the following tag, returning only '0' or '1':

Tag: "{tag}" -> 
""".strip()

    # Force short generation; we only want '0' or '1'.
    params = SamplingParams(
        temperature=0.2,
        max_tokens=2
    )
    outputs = llm.generate(prompt, params)
    response = outputs[0].outputs[0].text.strip()

    # Basic parse: if '1' is in the response => "useful", else => "not useful"
    # You can refine if you want to strictly check the first character, etc.
    if "1" in response:
        return "1"
    else:
        return "0"

def main():
    # 1) Load raw tags
    tags = pd.read_csv(os.path.join(RAW_DIR, "tags.csv"))
    print("Loaded tags.csv with shape:", tags.shape)

    # 2) Initialize vLLM with the Gemma model
    llm = LLM(
        model="google/gemma-3-27b-it",
        generation_config="vllm",
        tensor_parallel_size=4,
        max_model_len=8192
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
