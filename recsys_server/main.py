from fastapi import FastAPI
from recsys.router import router as recsys_router

app = FastAPI(title="RecSys Engine")

app.include_router(recsys_router, prefix="/api/recsys")
