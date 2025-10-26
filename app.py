"""
# Amazon Product Recommendation System API

## Structure
- User Management: Registration, authentication, JWT tokens
- Model Loading: Lazy loading with caching for all recommendation algorithms
- Recommendation Logic: Hybrid system with scenario detection (new/cold/warm users)
- API Endpoints: RESTful routes for recommendations, ratings, product details
- Metadata Handling: Product information enrichment with images and details

## Process Flow
1. User authenticates or uses guest mode
2. System detects user scenario based on rating history
3. Loads trained models (user-based, item-based, content-based, SVD, trending)
4. Applies adaptive weights based on user scenario
5. Generates hybrid recommendations
6. Enriches with product metadata
7. Returns personalized results with strategy info
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import polars as pl
import numpy as np
from pathlib import Path
import json
import pickle
from scipy.sparse import load_npz
from datetime import timedelta
import hashlib
import os

app = Flask(__name__)
CORS(app)

app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
jwt = JWTManager(app)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'

MODELS_CACHE = {}
METADATA_CACHE = {}

class UserDB:
    def __init__(self):
        self.users_file = DATA_DIR / 'users.json'
        self.users_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.users_file.exists():
            self.users_file.write_text(json.dumps({}))
    
    def load_users(self):
        return json.loads(self.users_file.read_text())
    
    def save_users(self, users):
        self.users_file.write_text(json.dumps(users, indent=2))
    
    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register(self, username, password, email, preferences=None):
        users = self.load_users()
        if username in users:
            return False, "Username already exists"
        users[username] = {
            'password': self.hash_password(password),
            'email': email,
            'preferences': preferences or {},
            'ratings': []
        }
        self.save_users(users)
        return True, "Registration successful"
    
    def authenticate(self, username, password):
        users = self.load_users()
        if username not in users:
            return False, "User not found"
        if users[username]['password'] != self.hash_password(password):
            return False, "Invalid password"
        return True, users[username]
    
    def get_user(self, username):
        return self.load_users().get(username)
    
    def update_rating_history(self, username, parent_asin, rating):
        users = self.load_users()
        if username in users:
            if 'rating_history' not in users[username]:
                users[username]['rating_history'] = []
            users[username]['rating_history'].append({
                'parent_asin': parent_asin,
                'rating': rating
            })
            self.save_users(users)

user_db = UserDB()

def load_hybrid_models(category):
    if category in MODELS_CACHE:
        return MODELS_CACHE[category]
    
    models = {}
    for algo in ['user', 'item', 'content', 'model', 'trending']:
        try:
            model_dir = MODELS_DIR / algo / category
            if not model_dir.exists():
                continue
            
            if algo == 'user':
                R = load_npz(model_dir / "R.npz")
                Rc = load_npz(model_dir / "Rc.npz") if (model_dir / "Rc.npz").exists() else None
                user_means = np.load(model_dir / "user_means.npy")
                with open(model_dir / "user_rev.pkl", "rb") as f:
                    user_rev = pickle.load(f)
                with open(model_dir / "item_rev.pkl", "rb") as f:
                    item_rev = pickle.load(f)
                user_idx = json.loads((model_dir / "user_idx.json").read_text())
                item_idx = json.loads((model_dir / "item_idx.json").read_text())
                with open(model_dir / "nn_model.pkl", "rb") as f:
                    nn_model = pickle.load(f)
                models['user'] = {
                    'R': R, 'Rc': Rc, 'user_means': user_means,
                    'user_rev': user_rev, 'item_rev': item_rev,
                    'user_idx': user_idx, 'item_idx': item_idx,
                    'nn_model': nn_model
                }
            
            elif algo == 'item':
                R = load_npz(model_dir / "R.npz")
                Rc = load_npz(model_dir / "Rc.npz") if (model_dir / "Rc.npz").exists() else None
                item_similarity = load_npz(model_dir / "item_similarity.npz")
                user_means = np.load(model_dir / "user_means.npy")
                with open(model_dir / "user_rev.pkl", "rb") as f:
                    user_rev = pickle.load(f)
                with open(model_dir / "item_rev.pkl", "rb") as f:
                    item_rev = pickle.load(f)
                user_idx = json.loads((model_dir / "user_idx.json").read_text())
                item_idx = json.loads((model_dir / "item_idx.json").read_text())
                models['item'] = {
                    'R': R, 'Rc': Rc, 'item_similarity': item_similarity,
                    'user_means': user_means, 'user_rev': user_rev,
                    'item_rev': item_rev, 'user_idx': user_idx,
                    'item_idx': item_idx
                }
            
            elif algo == 'content':
                R = load_npz(model_dir / "R.npz")
                item_similarity = load_npz(model_dir / "item_similarity.npz")
                with open(model_dir / "user_rev.pkl", "rb") as f:
                    user_rev = pickle.load(f)
                with open(model_dir / "item_rev.pkl", "rb") as f:
                    item_rev = pickle.load(f)
                user_idx = json.loads((model_dir / "user_idx.json").read_text())
                item_idx = json.loads((model_dir / "item_idx.json").read_text())
                models['content'] = {
                    'R': R, 'item_similarity': item_similarity,
                    'user_rev': user_rev, 'item_rev': item_rev,
                    'user_idx': user_idx, 'item_idx': item_idx
                }
            
            elif algo == 'model':
                R = load_npz(model_dir / "R.npz")
                U = np.load(model_dir / "U.npy")
                V = np.load(model_dir / "V.npy")
                with open(model_dir / "user_rev.pkl", "rb") as f:
                    user_rev = pickle.load(f)
                with open(model_dir / "item_rev.pkl", "rb") as f:
                    item_rev = pickle.load(f)
                user_idx = json.loads((model_dir / "user_idx.json").read_text())
                item_idx = json.loads((model_dir / "item_idx.json").read_text())
                models['model'] = {
                    'R': R, 'U': U, 'V': V,
                    'user_rev': user_rev, 'item_rev': item_rev,
                    'user_idx': user_idx, 'item_idx': item_idx
                }
            
            elif algo == 'trending':
                item_stats = pl.read_parquet(model_dir / 'item_stats.parquet')
                R = load_npz(model_dir / 'R.npz')
                with open(model_dir / "user_rev.pkl", "rb") as f:
                    user_rev = pickle.load(f)
                with open(model_dir / "item_rev.pkl", "rb") as f:
                    item_rev = pickle.load(f)
                user_idx = json.loads((model_dir / "user_idx.json").read_text())
                item_idx = json.loads((model_dir / "item_idx.json").read_text())
                models['trending'] = {
                    'item_stats': item_stats, 'R': R,
                    'user_rev': user_rev, 'item_rev': item_rev,
                    'user_idx': user_idx, 'item_idx': item_idx
                }
        except Exception as e:
            print(f"Failed to load {algo}: {e}")
    
    MODELS_CACHE[category] = models
    return models

def load_metadata(category):
    if category in METADATA_CACHE:
        return METADATA_CACHE[category]
    
    try:
        from configurations import Configurations
        processed_dir = Path(Configurations.DATA_PROCESSED_PATH)
    except:
        processed_dir = Path('data/processed')
    
    safe_cat = category.replace('/', '-')
    meta_file = processed_dir / f"{safe_cat}.meta.parquet"
    
    if meta_file.exists():
        df = pl.read_parquet(meta_file)
        METADATA_CACHE[category] = df
        return df
    return None

def detect_scenario(user_id, models, threshold=5):
    for algo in ['user', 'item', 'content', 'model']:
        if algo not in models:
            continue
        user_idx = models[algo].get('user_idx', {})
        if user_id not in user_idx:
            continue
        R = models[algo].get('R')
        if R is None:
            continue
        u = int(user_idx[user_id])
        n_ratings = R.getrow(u).nnz
        if n_ratings == 0:
            return 'new-user', 0
        elif n_ratings < threshold:
            return 'cold-user', n_ratings
        else:
            return 'warm-user', n_ratings
    return 'new-user', 0

def get_user_scenario_info(user_id, models):
    """Get detailed user scenario information"""
    scenario, n_ratings = detect_scenario(user_id, models, threshold=5)
    
    # Map scenario to readable labels
    scenario_labels = {
        'new-user': {
            'type': 'New User',
            'description': 'No rating history - showing popular trending items',
            'emoji': '🆕',
            'color': '#FF6B6B'
        },
        'cold-user': {
            'type': 'Cold User',
            'description': f'{n_ratings} ratings - combining trending with personalization',
            'emoji': '❄️',
            'color': '#FFD93D'
        },
        'warm-user': {
            'type': 'Warm User',
            'description': f'{n_ratings} ratings - personalized recommendations',
            'emoji': '🔥',
            'color': '#6BCF7F'
        }
    }
    
    # Active user (20+ ratings)
    if scenario == 'warm-user' and n_ratings > 20:
        return {
            'scenario': 'active-user',
            'type': 'Active User',
            'description': f'{n_ratings} ratings - highly personalized',
            'emoji': '⭐',
            'color': '#4ECDC4',
            'rating_count': n_ratings
        }
    
    info = scenario_labels.get(scenario, scenario_labels['new-user'])
    info['scenario'] = scenario
    info['rating_count'] = n_ratings
    return info

def get_item_scenario_contextual(parent_asin, models, user_scenario, strategy):
    """Get item scenario with context-aware labeling based on user scenario and strategy"""
    
    # Get basic item info
    for algo in ['user', 'item', 'content', 'model']:
        if algo not in models:
            continue
        
        R = models[algo].get('R')
        item_idx = models[algo].get('item_idx', {})
        
        if parent_asin not in item_idx:
            return {
                'scenario': 'new-item',
                'type': 'New Item',
                'description': 'Not in training set',
                'emoji': '🆕',
                'color': '#FF6B6B',
                'train_rating_count': 0
            }
        
        item_index = int(item_idx[parent_asin])
        train_rating_count = R.getcol(item_index).nnz
        
        # Context-aware labeling based on user scenario
        if user_scenario == 'new-user':
            # New user → Show trending items
            return {
                'scenario': 'trending',
                'type': '🔥 Trending',
                'description': f'Popular choice ({train_rating_count} ratings)',
                'emoji': '📈',
                'color': '#FF9900',
                'train_rating_count': train_rating_count
            }
        
        elif user_scenario in ['cold-user', 'warm-user', 'active-user']:
            # Personalized → Show recommendation reason
            if 'item' in strategy and train_rating_count > 20:
                return {
                    'scenario': 'personalized-cf',
                    'type': '🎯 For You',
                    'description': 'Based on your preferences',
                    'emoji': '💝',
                    'color': '#6BCF7F',
                    'train_rating_count': train_rating_count
                }
            elif 'content' in strategy:
                return {
                    'scenario': 'similar',
                    'type': '✨ Similar',
                    'description': 'Matches your interests',
                    'emoji': '🔍',
                    'color': '#4ECDC4',
                    'train_rating_count': train_rating_count
                }
            elif 'trending' in strategy:
                return {
                    'scenario': 'popular',
                    'type': '⭐ Popular',
                    'description': f'Highly rated ({train_rating_count} ratings)',
                    'emoji': '👍',
                    'color': '#FFD93D',
                    'train_rating_count': train_rating_count
                }
        
        # Default: Show item stats
        if train_rating_count == 0:
            scenario_info = {
                'scenario': 'new-item',
                'type': 'New',
                'description': '0 ratings',
                'emoji': '🆕',
                'color': '#FF6B6B'
            }
        elif train_rating_count <= 4:
            scenario_info = {
                'scenario': 'cold-item',
                'type': 'Emerging',
                'description': f'{train_rating_count} ratings',
                'emoji': '🌱',
                'color': '#FFD93D'
            }
        elif train_rating_count <= 20:
            scenario_info = {
                'scenario': 'warm-item',
                'type': 'Popular',
                'description': f'{train_rating_count} ratings',
                'emoji': '📦',
                'color': '#6BCF7F'
            }
        else:
            scenario_info = {
                'scenario': 'popular-item',
                'type': 'Best Seller',
                'description': f'{train_rating_count} ratings',
                'emoji': '🔥',
                'color': '#4ECDC4'
            }
        
        scenario_info['train_rating_count'] = train_rating_count
        return scenario_info
    
    return {
        'scenario': 'unknown',
        'type': 'Unknown',
        'description': 'Not in model',
        'emoji': '❓',
        'color': '#999999',
        'train_rating_count': 0
    }

def predict_user(user_id, models, k=30):
    if 'user' not in models:
        return None
    
    art = models['user']
    
    if user_id not in art.get('user_idx', {}):
        return None
    
    try:
        u = int(art['user_idx'][user_id])
        X = art['Rc'] if art.get('Rc') is not None else art['R']
        nn_model = art.get('nn_model')
        
        if nn_model is None:
            print(f"ERROR: nn_model is None for user {user_id}")
            return None
        
        if not hasattr(nn_model, 'kneighbors'):
            print(f"ERROR: nn_model is {type(nn_model)}, expected NearestNeighbors")
            return None
        
        distances, indices = nn_model.kneighbors(X.getrow(u), return_distance=True)
        d, idx = distances.ravel(), indices.ravel()
        mask = idx != u
        idx, d = idx[mask][:k], d[mask][:k]
        
        if idx.size == 0:
            return None
        
        sims = np.clip(1.0 - d, 0.0, 1.0)
        scores = X[idx, :].T.dot(sims) / (np.sum(np.abs(sims)) + 1e-8)
        
        if art.get('Rc') is not None:
            scores = scores + art['user_means'][u]
        
        return scores
    except Exception as e:
        print(f"ERROR in predict_user for {user_id}: {e}")
        return None

def predict_item(user_id, models, k=30):
    if 'item' not in models or user_id not in models['item']['user_idx']:
        return None
    art = models['item']
    u = int(art['user_idx'][user_id])
    R, Rc = art['R'], art.get('Rc')
    X = Rc if Rc is not None else R
    user_ratings = X.getrow(u).toarray().ravel()
    rated_items = np.nonzero(R.getrow(u).toarray().ravel())[0]
    if len(rated_items) == 0:
        return None
    item_sim = art['item_similarity']
    scores = np.zeros(R.shape[1], dtype=np.float32)
    for i in range(R.shape[1]):
        sims = item_sim[i, rated_items].toarray().ravel()
        top_idx = np.argpartition(-sims, min(k, len(sims)))[:k] if len(sims) > k else np.arange(len(sims))
        top_sims, top_ratings = sims[top_idx], user_ratings[rated_items[top_idx]]
        if np.sum(np.abs(top_sims)) > 1e-8:
            scores[i] = np.sum(top_sims * top_ratings) / np.sum(np.abs(top_sims))
    if Rc is not None:
        scores = scores + art['user_means'][u]
    return scores

def predict_content(user_id, models, k=30):
    if 'content' not in models or user_id not in models['content']['user_idx']:
        return None
    art = models['content']
    u = int(art['user_idx'][user_id])
    R = art['R']
    user_ratings = R.getrow(u).toarray().ravel()
    rated_items = np.nonzero(user_ratings)[0]
    if len(rated_items) == 0:
        return None
    item_sim = art['item_similarity']
    scores = np.zeros(R.shape[1], dtype=np.float32)
    for i in range(R.shape[1]):
        sims = item_sim[i, rated_items].toarray().ravel()
        top_idx = np.argpartition(-sims, min(k, len(sims)))[:k] if len(sims) > k else np.arange(len(sims))
        top_sims, top_ratings = sims[top_idx], user_ratings[rated_items[top_idx]]
        if np.sum(np.abs(top_sims)) > 1e-8:
            scores[i] = np.sum(top_sims * top_ratings) / np.sum(np.abs(top_sims))
    return scores

def predict_model(user_id, models):
    if 'model' not in models or user_id not in models['model']['user_idx']:
        return None
    art = models['model']
    u = int(art['user_idx'][user_id])
    return art['U'][u] @ art['V'].T

def predict_trending(user_id, models):
    if 'trending' not in models:
        return None
    art = models['trending']
    item_stats = art['item_stats']
    item_idx = art['item_idx']
    scores = np.zeros(len(item_idx), dtype=np.float32)
    top_trending = item_stats.head(100)
    for rank, row in enumerate(top_trending.iter_rows(named=True)):
        item = row['parent_asin']
        if item in item_idx:
            idx = int(item_idx[item])
            scores[idx] = 1.0 / (rank + 1)
    return scores

def predict_hybrid(user_id, models, weights=None):
    scenario, n_ratings = detect_scenario(user_id, models, threshold=5)
    
    predictions = {}
    for name, pred_fn in [('user', predict_user), ('item', predict_item),
                          ('content', predict_content), ('model', predict_model),
                          ('trending', predict_trending)]:
        scores = pred_fn(user_id, models)
        if scores is not None:
            predictions[name] = scores
    
    if not predictions:
        return None, "no_models"
    
    weight_configs = {
        'new-user': {'user': 0.0, 'item': 0.0, 'content': 0.0, 'model': 0.0, 'trending': 1.0},
        'cold-user': {'user': 0.1, 'item': 0.1, 'content': 0.3, 'model': 0.1, 'trending': 0.4},
        'warm-user': {'user': 0.25, 'item': 0.35, 'content': 0.20, 'model': 0.20, 'trending': 0.0}
    }
    
    available = list(predictions.keys())
    base_weights = weight_configs.get(scenario, weight_configs['warm-user'])
    adaptive_weights = {m: w for m, w in base_weights.items() if m in available}
    
    if scenario == 'cold-user' and n_ratings >= 3:
        if 'user' in adaptive_weights:
            adaptive_weights['user'] *= 1.5
        if 'item' in adaptive_weights:
            adaptive_weights['item'] *= 1.5
        if 'trending' in adaptive_weights:
            adaptive_weights['trending'] *= 0.7
    elif scenario == 'warm-user' and n_ratings > 20:
        if 'user' in adaptive_weights:
            adaptive_weights['user'] *= 1.2
        if 'item' in adaptive_weights:
            adaptive_weights['item'] *= 1.2
    
    total = sum(adaptive_weights.values())
    if total > 0:
        adaptive_weights = {k: v / total for k, v in adaptive_weights.items()}
    
    ref_model = available[0]
    n_items = len(predictions[ref_model])
    combined = np.zeros(n_items, dtype=np.float32)
    
    for model_name, scores in predictions.items():
        weight = adaptive_weights.get(model_name, 0.0)
        if weight > 0:
            combined += weight * scores
    
    parts = [f"{m}({w*100:.0f}%)" for m, w in sorted(adaptive_weights.items(), key=lambda x: -x[1]) if w > 0.05]
    strategy = f"hybrid-{scenario}-" + "+".join(parts)
    
    return combined, strategy

def get_recommendations(user_id, category, n_recs=10):
    models = load_hybrid_models(category)
    if not models:
        return [], "no_models"
    
    scores, strategy = predict_hybrid(user_id, models)
    
    if scores is None:
        return [], "no_predictions"
    
    for algo in ['user', 'item', 'content', 'model']:
        if algo in models:
            R = models[algo]['R']
            item_rev = models[algo]['item_rev']
            item_idx = models[algo]['item_idx']
            user_idx = models[algo]['user_idx']
            if user_id in user_idx:
                u = int(user_idx[user_id])
                rated = set(R.getrow(u).indices.tolist())
            else:
                rated = set()
            
            # Add dynamically rated items from rating_history
            user_data = user_db.get_user(user_id)
            if user_data and 'rating_history' in user_data:
                for rating_record in user_data['rating_history']:
                    rated_asin = rating_record['parent_asin']
                    if rated_asin in item_idx:
                        rated.add(int(item_idx[rated_asin]))
            break
    
    cand_mask = np.ones(len(scores), dtype=bool)
    if rated:
        cand_mask[list(rated)] = False
    
    cand_scores = scores[cand_mask]
    if cand_scores.size == 0:
        return [], "no_candidates"
    
    n_top = min(n_recs, cand_scores.size)
    cand_indices = np.nonzero(cand_mask)[0]
    top_pos = np.argpartition(-cand_scores, n_top - 1)[:n_top]
    top_pos = top_pos[np.argsort(-cand_scores[top_pos])]
    
    recommendations = [(item_rev[cand_indices[i]], float(cand_scores[i])) for i in top_pos]
    return recommendations, strategy

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')
    
    if not all([username, password, email]):
        return jsonify({'error': 'Missing fields'}), 400
    
    success, message = user_db.register(username, password, email)
    if success:
        return jsonify({'message': message}), 201
    return jsonify({'error': message}), 400

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not all([username, password]):
        return jsonify({'error': 'Missing credentials'}), 400
    
    success, result = user_db.authenticate(username, password)
    if success:
        access_token = create_access_token(identity=username)
        return jsonify({
            'access_token': access_token,
            'username': username,
            'email': result['email']
        }), 200
    return jsonify({'error': result}), 401

@app.route('/api/recommendations/<category>', methods=['GET'])
@jwt_required(optional=True)
def get_user_recommendations(category):
    username = get_jwt_identity()
    if username is None:
        username = 'guest'
    n_recs = request.args.get('n', default=10, type=int)
    
    try:
        models = load_hybrid_models(category)
        recommendations, strategy = get_recommendations(username, category, n_recs)
        meta_df = load_metadata(category)
        results = []
        
        # Get user scenario FIRST
        user_scenario = get_user_scenario_info(username, models)
        user_scenario_type = user_scenario['scenario']  # 'new-user', 'cold-user', 'warm-user', 'active-user'
        
        for parent_asin, score in recommendations:
            item_data = {
                'parent_asin': parent_asin,
                'score': float(score),
                'title': 'Product information loading...',
                'price': 'N/A',
                'rating': 0,
                'rating_number': 0,
                'image_url': None
            }
            
            if meta_df is not None:
                meta_row = meta_df.filter(pl.col('parent_asin') == parent_asin)
                if len(meta_row) > 0:
                    row_dict = meta_row.to_dicts()[0]
                    item_data['title'] = row_dict.get('title', 'Unknown Product')
                    price_raw = row_dict.get('price', 'N/A')
                    if price_raw and price_raw != 'N/A':
                        item_data['price'] = price_raw
                    item_data['rating'] = float(row_dict.get('average_rating', 0) or 0)
                    item_data['rating_number'] = int(row_dict.get('rating_number', 0) or 0)
                    
                    images = row_dict.get('images', [])
                    if isinstance(images, list) and len(images) > 0:
                        img_obj = images[0]
                        if isinstance(img_obj, dict):
                            item_data['image_url'] = img_obj.get('hi_res') or img_obj.get('large') or img_obj.get('thumb')
                        elif isinstance(img_obj, str):
                            item_data['image_url'] = img_obj
                    
                    features = row_dict.get('features', [])
                    if isinstance(features, list):
                        item_data['features'] = features[:5]
                    
                    desc = row_dict.get('description', [])
                    if isinstance(desc, list) and len(desc) > 0:
                        item_data['description'] = desc[0][:500]
            
            results.append(item_data)
        
        # Add item scenario info with context-aware labeling
        for item_data in results:
            item_scenario = get_item_scenario_contextual(
                item_data['parent_asin'], 
                models, 
                user_scenario_type,
                strategy
            )
            item_data['item_scenario'] = item_scenario
        
        return jsonify({
            'recommendations': results,
            'strategy': strategy,
            'count': len(results),
            'user': username,
            'category': category,
            'user_scenario': user_scenario
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    categories = []
    if MODELS_DIR.exists():
        for algo_dir in MODELS_DIR.iterdir():
            if algo_dir.is_dir():
                for cat_dir in algo_dir.iterdir():
                    if cat_dir.is_dir() and cat_dir.name not in categories:
                        categories.append(cat_dir.name)
    return jsonify({'categories': list(set(categories))}), 200

@app.route('/api/rate', methods=['POST'])
@jwt_required()
def rate_product():
    username = get_jwt_identity()
    data = request.json
    parent_asin = data.get('parent_asin')
    rating = data.get('rating')
    
    if not all([parent_asin, rating]):
        return jsonify({'error': 'Missing fields'}), 400
    
    user_db.update_rating_history(username, parent_asin, rating)
    return jsonify({'message': 'Rating recorded'}), 200

@app.route('/api/product/<parent_asin>', methods=['GET'])
def get_product_details(parent_asin):
    category = request.args.get('category', 'Electronics')
    meta_df = load_metadata(category)
    
    if meta_df is not None:
        product = meta_df.filter(pl.col('parent_asin') == parent_asin)
        if len(product) > 0:
            product_dict = product.to_dicts()[0]
            images = product_dict.get('images', [])
            if isinstance(images, list):
                product_dict['images'] = [img.get('large', img.get('thumb', '')) for img in images if isinstance(img, dict)]
            return jsonify(product_dict), 200
    return jsonify({'error': 'Product not found'}), 404

@app.route('/api/placeholder-image', methods=['GET'])
def placeholder_image():
    svg = '''<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
        <rect width="200" height="200" fill="#f0f0f0"/>
        <text x="50%" y="50%" text-anchor="middle" fill="#999" font-size="16">No Image</text>
    </svg>'''
    return svg, 200, {'Content-Type': 'image/svg+xml'}

@app.route('/api/cold-items/<category>', methods=['GET'])
def get_cold_items(category):
    try:
        print(f"Loading cold items for category: {category}")
        
        # Load train data directly
        try:
            from configurations import Configurations
            processed_dir = Path(Configurations.DATA_PROCESSED_PATH)
        except:
            processed_dir = Path('data/processed')
        
        safe_cat = category.replace('/', '-')
        train_file = processed_dir / f"{safe_cat}.5core.train.parquet"
        
        if not train_file.exists():
            print(f"ERROR: Train file not found: {train_file}")
            return jsonify({'error': 'Train data not found', 'grouped_items': {}}), 200
        
        print(f"Loading train data from: {train_file}")
        train = pl.read_parquet(train_file)
        
        # Count ratings per item
        item_counts = train.group_by('parent_asin').agg(pl.len().alias('train_count'))
        
        # Group items by rating count (1, 2, 3, 4) - max 10 items each
        grouped_items = {}
        meta_df = load_metadata(category)
        
        for rating_level in [1, 2, 3, 4]:
            items_at_level = item_counts.filter(
                pl.col('train_count') == rating_level
            ).head(20)  # Changed from 10 to 20 to have buffer for refills
            
            items_list = []
            for row in items_at_level.iter_rows(named=True):
                item_data = {
                    'parent_asin': row['parent_asin'],
                    'train_rating_count': row['train_count'],
                    'title': f"Item {row['parent_asin'][:8]}",
                    'rating': 0,
                    'image_url': None
                }
                
                # Enrich with metadata
                if meta_df is not None:
                    meta_row = meta_df.filter(pl.col('parent_asin') == row['parent_asin'])
                    if len(meta_row) > 0:
                        row_dict = meta_row.to_dicts()[0]
                        item_data['title'] = row_dict.get('title', item_data['title'])
                        item_data['rating'] = float(row_dict.get('average_rating', 0) or 0)
                        images = row_dict.get('images', [])
                        if isinstance(images, list) and len(images) > 0:
                            img_obj = images[0]
                            if isinstance(img_obj, dict):
                                # FIXED: Use hi_res -> large -> thumb (same priority as recommendations)
                                item_data['image_url'] = img_obj.get('hi_res') or img_obj.get('large') or img_obj.get('thumb')
                            elif isinstance(img_obj, str):
                                item_data['image_url'] = img_obj
                
                items_list.append(item_data)
            
            grouped_items[str(rating_level)] = items_list
            print(f"  {rating_level}-rating: {len(items_list)} items")
        
        total_items = sum(len(items) for items in grouped_items.values())
        print(f"Returning {total_items} cold items grouped by rating")
        
        return jsonify({
            'grouped_items': grouped_items,
            'category': category
        }), 200
    except Exception as e:
        print(f"ERROR in get_cold_items: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'grouped_items': {}}), 200

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'models_loaded': len(MODELS_CACHE)}), 200

if __name__ == '__main__':
    print("Starting Amazon Product Recommendation System API")
    print(f"Models: {MODELS_DIR}")
    print(f"Data: {DATA_DIR}")
    app.run(debug=True, host='0.0.0.0', port=5000)