import os
from dotenv import load_dotenv

load_dotenv()

HMAC_SECRET = os.getenv("HMAC_SECRET", "super-secret")
DB_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/yourdb")
