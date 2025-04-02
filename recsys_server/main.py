from fastapi import FastAPI
from recsys.router import router as recsys_router
from recsys.data.feature_store import init_feature_store

app = FastAPI(title="RecSys Engine")

@app.on_event("startup")
async def startup_event():
    # Load static item data into memory and compute preprocessed features.
    init_feature_store()

app.include_router(recsys_router, prefix="/api/recsys")
