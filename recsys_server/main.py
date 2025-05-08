from fastapi import FastAPI
from recsys.router import router as recsys_router
from recsys.data.feature_store import init_feature_store
from recsys.logic.two_tower import init_two_tower_model

app = FastAPI(title="RecSys Engine")

@app.on_event("startup")
async def startup_event():
    # Load static item data into memory and compute preprocessed features.
    print("Loading static item data into memory and computing preprocessed features...")
    init_feature_store()
    print("Static item data loaded and preprocessed features computed.")
    # Load the two-tower model into memory.
    print("Loading TwoTower model into memory...")
    init_two_tower_model()
    print("TwoTower model loaded into memory.")

app.include_router(recsys_router, prefix="/api/recsys")
