# Canteen-Recommendation-System
# ML Module — Canteen Recommendation System
## API (recommendation)
https://canteen-recommendation-system.onrender.com/docs
## API (chatbot)
https://canteen-recommendation-system.onrender.com
## Endpoints
### ✅ 1. Global popularity
GET /recommend?top_n=5&window_days=7

### ✅ 2. Personalized (if model exists)
GET /recommend/user/{user_id}

Fallback → global popular

---

## Train Phase 2 Model
```bash
python ML/Model/train_recommender.py


