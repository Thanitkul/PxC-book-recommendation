#!/usr/bin/env python3
# fast_goodreads_dlrm_batch_inference_mp_merged.py
# ------------------------------------------------------------
# 10+ model instances – parallel – writes ONE output file + summary
# ------------------------------------------------------------
import os, sys, json, time
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from typing import Dict, List

# ───────── CONFIG ─────────
DATA_ROOT   = "../data-prep-EDA/clean"
CKPT_PATH   = "trained_dlrm_goodreads.pt"
OUT_DIR     = "inference_outputs"
MERGED_TXT  = "goodreads_batch_inference.txt"
SUMMARY_JSON = "summary.json"
DEVICE_ID   = 0
BATCH_SIZE  = 8192
NUM_WORKERS = 10
TOP_K       = 10

# ───────── Utilities ─────────
def sys_path_insert_once(p:str):
    if p not in sys.path: sys.path.insert(0, p)

def chunk(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i*k+min(i,m):(i+1)*k+min(i+1,m)] for i in range(n)]

# ───────── Feature Loader ─────────
def load_artifacts(data_root:str, ckpt_path:str, device:torch.device) -> Dict:
    csv = lambda name: os.path.join(data_root, name)
    books      = pd.read_csv(csv("books.csv"))
    ratings    = pd.read_csv(csv("ratings.csv"))
    wish_train = pd.read_csv(csv("to_read_train.csv"))
    wish_test  = pd.read_csv(csv("to_read_test.csv"))
    book_tags  = pd.read_csv(csv("book_tags.csv"))
    tags_df    = pd.read_csv(csv("tags.csv"))

    author2idx = {a:i+1 for i,a in enumerate(sorted(books.authors.unique()))}
    lang2idx   = {l:i+1 for i,l in enumerate(books.language_code.fillna("unk").unique())}

    top_tags = {
        bid: (grp.sort_values("count",ascending=False).tag_id.tolist()[:5] + [0]*5)[:5]
        for bid,grp in book_tags.groupby("book_id")
    }

    tag_id2name = tags_df.set_index("tag_id")["tag_name"].to_dict()

    book_author = books.set_index("book_id").authors.map(author2idx.get).fillna(0).astype(int).to_dict()
    book_lang   = books.set_index("book_id").language_code.fillna("unk").map(lang2idx.get).fillna(0).astype(int).to_dict()
    book_dense  = books.set_index("book_id")[["ratings_count","average_rating"]].astype(float).to_dict()

    ratings_by_user = ratings.groupby("user_id").book_id.apply(list).to_dict()
    wish_by_user    = wish_test.groupby("user_id").book_id.apply(list).to_dict()

    all_books = books.book_id.values.astype(np.int64)
    max_rc    = float(books.ratings_count.max() or 1)

    sys_path_insert_once("dlrm")
    from dlrm_s_pytorch import DLRM_Net
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = DLRM_Net(
        m_spa       = ckpt["embed_dim"],
        ln_emb      = np.asarray(ckpt["embedding_sizes"], dtype=np.int64),
        ln_bot      = np.asarray(ckpt["bottom_mlp"],      dtype=np.int64),
        ln_top      = np.asarray(ckpt["top_mlp"],         dtype=np.int64),
        arch_interaction_op="dot",
        sigmoid_bot = -1,
        sigmoid_top = len(ckpt["top_mlp"])-2,
        ndevices    = -1,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return dict(
        model=model, device=device,
        books=books,
        tag_id2name=tag_id2name,
        ratings_by_user=ratings_by_user,
        wish_by_user=wish_by_user,
        all_books=all_books,
        book_author=book_author, book_lang=book_lang, book_dense=book_dense,
        top_tags=top_tags, max_rc=max_rc
    )

# ───────── Single Worker ─────────
def worker_process(user_ids:List[int], cfg:Dict, return_dict, rank:int):
    torch.cuda.set_device(cfg["device"])
    art   = load_artifacts(DATA_ROOT, CKPT_PATH, cfg["device"])
    model = art["model"]; device = art["device"]
    PAD = 0
    books = art["all_books"]

    output = []

    for uid in user_ids:
        rated = art["ratings_by_user"].get(uid, [])[:20]
        wish  = art["wish_by_user"]  .get(uid, [])[:20]
        rated_pad = rated + [PAD]*(20-len(rated))
        wish_pad  = wish + [PAD]*(20-len(wish))

        def title_and_tags(bid:int) -> str:
            title = art["books"].loc[art["books"].book_id==bid, "title"].values[0]
            tags  = [art["tag_id2name"].get(tid,"") for tid in art["top_tags"].get(bid,[]) if tid][:5]
            return f"{title} [Tags: {', '.join(tags)}]"

        rated_lines = [title_and_tags(bid) for bid in rated]
        wish_lines  = [title_and_tags(bid) for bid in wish]

        # favorite genres
        tag_counter = {}
        for bid in rated+wish:
            for tid in art["top_tags"].get(bid, []):
                if tid: tag_counter[tid] = tag_counter.get(tid, 0) + 1
        fav_tags = sorted(tag_counter, key=tag_counter.get, reverse=True)[:5]
        fav_genres = [art["tag_id2name"].get(tid,"") for tid in fav_tags]

        # forward pass
        rated_t = torch.tensor(rated_pad, dtype=torch.long, device=device)
        wish_base = torch.tensor(wish_pad, dtype=torch.long, device=device)

        top_prob  = torch.full((TOP_K,), -1.0,  device="cpu")
        top_book  = torch.full((TOP_K,), -1,     dtype=torch.long, device="cpu")

        for b_start in range(0, len(books), BATCH_SIZE):
            cand     = books[b_start:b_start+BATCH_SIZE]
            C        = len(cand)
            wish_block = wish_base.repeat(C,1)
            wish_block[:,-1] = torch.tensor(cand, dtype=torch.long, device=device)
            base_sparse = torch.cat([rated_t.repeat(C,1), wish_block], dim=1)

            aux = torch.tensor(
                [[art["book_author"].get(int(b),0),
                  art["book_lang"].get(int(b),0),
                 *art["top_tags"].get(int(b),[0]*5)] for b in cand],
                dtype=torch.long, device=device)
            sparse47 = torch.cat([base_sparse, aux], dim=1)

            lS_i = [sparse47[:,i] for i in range(47)]
            offs = torch.arange(C, dtype=torch.long, device=device)
            lS_o = [offs]*47

            recs = art["book_dense"]
            dense_np = np.zeros((C,2), dtype=np.float32)
            dense_np[:,0] = np.clip(
                np.log1p([recs.get(int(b),{"ratings_count":0})["ratings_count"] for b in cand])
                / np.log1p(art["max_rc"]), 0, 1)
            dense_np[:,1] = [(recs.get(int(b),{"average_rating":0})["average_rating"]-1)/4 for b in cand]
            dense2 = torch.tensor(dense_np, dtype=torch.float32, device=device)

            with torch.no_grad():
                prob = model(dense2, lS_o, lS_i).squeeze(1).cpu()

            merged_p   = torch.cat([top_prob, prob])
            merged_idx = torch.cat([top_book, torch.from_numpy(cand)])
            sel = torch.topk(merged_p, TOP_K)
            top_prob, top_book = sel.values, merged_idx[sel.indices]

        # prepare output for this user
        user_txt = []
        user_txt.append(f"User ID: {uid}")
        user_txt.append("\nRated Books:")
        for line in rated_lines:
            user_txt.append(f"  - {line}")
        user_txt.append("\nWishlist Books:")
        for line in wish_lines:
            user_txt.append(f"  - {line}")
        user_txt.append("\nFavorite Genres:")
        user_txt.append("  " + ", ".join(fav_genres))

        user_txt.append("\nTop-10 Recommendations:")
        order = torch.argsort(-top_prob)
        for b, p in zip(top_book[order].tolist(), top_prob[order].tolist()):
            rec_title = art["books"].loc[art["books"].book_id==b, "title"].values[0]
            rec_tags  = [art["tag_id2name"].get(tid,"") for tid in art["top_tags"].get(b,[]) if tid][:5]
            user_txt.append(f"  - {rec_title} (prob={p:.4f}) | Genres: {', '.join(rec_tags)}")

        output.append("\n".join(user_txt) + "\n" + "="*80 + "\n")

    return_dict[rank] = output

# ───────── orchestrator ─────────
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    dummy_art = load_artifacts(DATA_ROOT, CKPT_PATH, torch.device(f"cuda:{DEVICE_ID}"))
    user_list = sorted(dummy_art["wish_by_user"].keys())
    user_chunks = chunk(user_list, NUM_WORKERS)

    print(f"🚀 Spawning {NUM_WORKERS} workers on cuda:{DEVICE_ID} …")
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_dict = manager.dict()

    t0 = time.time()
    procs = []
    for rk, chunk_ids in enumerate(user_chunks):
        cfg = dict(device=torch.device(f"cuda:{DEVICE_ID}"))
        p = mp.Process(target=worker_process, args=(chunk_ids, cfg, result_dict, rk), daemon=True)
        p.start()
        procs.append(p)
    for p in procs: p.join()
    total = time.time() - t0

    all_texts = []
    for outputs in result_dict.values():
        all_texts.extend(outputs)

    with open(os.path.join(OUT_DIR, MERGED_TXT), "w", encoding="utf-8") as f:
        f.writelines(all_texts)

    total_users = len(user_list)
    summary = {
        "total_users": total_users,
        "total_wall_time_sec": total,
        "average_time_per_user_sec": total/total_users,
        "requests_per_second": total_users/total
    }
    with open(os.path.join(OUT_DIR, SUMMARY_JSON), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Finished {total_users} users.")
    print(f"Summary JSON saved: {SUMMARY_JSON}")
