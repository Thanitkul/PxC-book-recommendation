
# PxC Book Recommendation System 📚

An end-to-end, modular recommendation system for books, combining:
- ✅ Cold-start model (content-based)
- ✅ Two-tower deep learning model (history + wishlist)
- ✅ User-based collaborative filtering (kNN)

Supports secure REST API access (HTTPS) and Angular frontend clients for both users and admins.

---

## 🔧 Technologies

- **Backend**: FastAPI (Python)
- **Recommendation Models**: PyTorch, NumPy, SciKit-Learn
- **Database**: PostgreSQL 15
- **Frontend**: Angular (User/Admin Apps)
- **Deployment**: Docker Compose

---


## 🚀 How to Run

> ⚠️ Requires: Docker + Docker Compose

1. **Add model checkpoint**  
   Place your trained model at:
```

recsys\_server/recsys/models/two\_tower\_pointwise\_bce\_prefilter\_80.pt

````

2. **Build & launch everything**  
From the root directory:
```bash
docker compose up --build
````

3. **Run dataloader** *(once, manually)*
   After all services are up:

   ```bash
   docker compose run --rm dataloader
   ```

4. **Visit the apps**

   * User app: [http://localhost:4201](http://localhost:4201)
   * Admin app: [http://localhost:4200](http://localhost:4200)
   * API: [http://localhost:8080](http://localhost:8080) (HTTP) or [https://localhost:8443](https://localhost:8443) (HTTPS)

---

## 🔁 Recommendation Flow

1. **Cold-start**: Used when user has no ratings
2. **Main models** (for active users):

   * Two-tower: recommends 70 books
   * Collaborative: recommends 30 books
3. **Final output**: Top 100 unique books, excluding rated/wishlisted ones

---

## 🧪 Testing

To test recommendations directly via REST API:

```bash
curl -X POST http://localhost:8080/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 123}'
```

Response:

```json
["book_id_1", "book_id_2", ...]
```


## 👨‍🏫 Project Notes

This system was built as part of an academic assignment to demonstrate:

* Multi-model hybrid recommendation
* Full-stack Dockerized deployment
* FastAPI backend with secure API design
* Data pipeline and schema management

