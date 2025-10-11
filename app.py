from flask import Flask, request, jsonify, send_file
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

class SimpleLogger:
    def log_info(self, msg):
        print(f"[INFO] {msg}")
    def log_warning(self, msg):
        print(f"[WARNING] {msg}")
    def log_exception(self, msg):
        print(f"[ERROR] {msg}")

logger = SimpleLogger()

app = Flask(__name__)
CORS(app)

# Configuration
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
jwt = JWTManager(app)

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
IMAGES_DIR = DATA_DIR / 'images'

# Global cache for models
MODELS_CACHE = {}

# ============= USER MANAGEMENT =============

class UserDB:
    """Simple user database (in production, use proper DB)"""
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
            'rating_history': []
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
        users = self.load_users()
        return users.get(username)
    
    def update_rating_history(self, username, parent_asin, rating):
        users = self.load_users()
        if username in users:
            users[username]['rating_history'].append({
                'parent_asin': parent_asin,
                'rating': rating
            })
            self.save_users(users)

user_db = UserDB()

# ============= MODEL LOADING =============

def load_hybrid_models(category: str):
    """Load all models for hybrid recommendation"""
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
                with open(model_dir / "user_rev.pkl", "rb") as f: user_rev = pickle.load(f)
                with open(model_dir / "item_rev.pkl", "rb") as f: item_rev = pickle.load(f)
                user_idx = json.loads((model_dir / "user_idx.json").read_text())
                item_idx = json.loads((model_dir / "item_idx.json").read_text())
                with open(model_dir / "nn_model.pkl", "rb") as f: nn_model = pickle.load(f)
                models['user'] = {'R': R, 'Rc': Rc, 'user_means': user_means, 'user_rev': user_rev, 
                                 'item_rev': item_rev, 'user_idx': user_idx, 'item_idx': item_idx, 'nn_model': nn_model}
            
            elif algo == 'item':
                R = load_npz(model_dir / "R.npz")
                Rc = load_npz(model_dir / "Rc.npz") if (model_dir / "Rc.npz").exists() else None
                item_similarity = load_npz(model_dir / "item_similarity.npz")
                user_means = np.load(model_dir / "user_means.npy")
                with open(model_dir / "user_rev.pkl", "rb") as f: user_rev = pickle.load(f)
                with open(model_dir / "item_rev.pkl", "rb") as f: item_rev = pickle.load(f)
                user_idx = json.loads((model_dir / "user_idx.json").read_text())
                item_idx = json.loads((model_dir / "item_idx.json").read_text())
                models['item'] = {'R': R, 'Rc': Rc, 'item_similarity': item_similarity, 'user_means': user_means,
                                 'user_rev': user_rev, 'item_rev': item_rev, 'user_idx': user_idx, 'item_idx': item_idx}
            
            elif algo == 'content':
                R = load_npz(model_dir / "R.npz")
                item_similarity = load_npz(model_dir / "item_similarity.npz")
                with open(model_dir / "user_rev.pkl", "rb") as f: user_rev = pickle.load(f)
                with open(model_dir / "item_rev.pkl", "rb") as f: item_rev = pickle.load(f)
                user_idx = json.loads((model_dir / "user_idx.json").read_text())
                item_idx = json.loads((model_dir / "item_idx.json").read_text())
                models['content'] = {'R': R, 'item_similarity': item_similarity, 'user_rev': user_rev,
                                    'item_rev': item_rev, 'user_idx': user_idx, 'item_idx': item_idx}
            
            elif algo == 'model':
                R = load_npz(model_dir / "R.npz")
                U = np.load(model_dir / "U.npy")
                V = np.load(model_dir / "V.npy")
                with open(model_dir / "user_rev.pkl", "rb") as f: user_rev = pickle.load(f)
                with open(model_dir / "item_rev.pkl", "rb") as f: item_rev = pickle.load(f)
                user_idx = json.loads((model_dir / "user_idx.json").read_text())
                item_idx = json.loads((model_dir / "item_idx.json").read_text())
                models['model'] = {'R': R, 'U': U, 'V': V, 'user_rev': user_rev, 
                                  'item_rev': item_rev, 'user_idx': user_idx, 'item_idx': item_idx}
            
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
                    'item_stats': item_stats,
                    'R': R,
                    'user_rev': user_rev,
                    'item_rev': item_rev,
                    'user_idx': user_idx,
                    'item_idx': item_idx
                }
        
        except Exception as e:
            print(f"Failed to load {algo} model: {e}")
    
    MODELS_CACHE[category] = models
    return models

def predict_trending(user_id: str, models: dict):
    """Get trending scores as prediction array"""
    if 'trending' not in models:
        return None
    
    art = models['trending']
    item_stats = art['item_stats']
    item_idx = art['item_idx']
    
    # Create scores array
    scores = np.zeros(len(item_idx), dtype=np.float32)
    
    # Get top 100 trending items
    top_trending = item_stats.head(100)
    
    for rank, row in enumerate(top_trending.iter_rows(named=True)):
        item = row['parent_asin']
        if item in item_idx:
            idx = int(item_idx[item])
            # Reciprocal rank scoring
            scores[idx] = 1.0 / (rank + 1)
    
    return scores

def detect_scenario(user_id: str, models: dict, threshold: int = 5):
    """
    Detect user scenario: new-user, cold-user, warm-user
    Returns: (scenario, n_ratings)
    """
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

# Add for quick loading metadata
METADATA_CACHE = {}

def load_metadata(category):
    """Load metadata với cache, chỉ cho items trong training data"""
    # Check cache
    if category in METADATA_CACHE:
        return METADATA_CACHE[category]
    
    try:
        from configurations import Configurations
        processed_dir = Path(Configurations.DATA_PROCESSED_PATH)
    except:
        processed_dir = Path('/Users/kevin/Documents/GitHub/Python/VESKL/Personal/NEU/NEU/NEU_7275/Prj/Amazon-Product-Recommendation-System/data/processed')
    
    print("processed_dir:", processed_dir)
    safe_cat = category.replace('/', '-')
    meta_file = processed_dir / f"{safe_cat}.meta.parquet"
    
    if meta_file.exists():
        print(f"Loading metadata: {meta_file.name} ({meta_file.stat().st_size / 1024 / 1024:.1f} MB)...")
        df = pl.read_parquet(meta_file)
        
        # Cache để lần sau nhanh
        METADATA_CACHE[category] = df
        
        print(f"✓ Loaded {len(df):,} products")
        return df
    
    print(f"Metadata not found: {meta_file}")
    return None

# ============= RECOMMENDATION LOGIC =============

def detect_cold_start(user_id: str, models: dict, min_ratings: int = 5):
    """Check if user is new or has few ratings"""
    for algo in ['user', 'item', 'model']:
        if algo in models and user_id in models[algo]['user_idx']:
            R = models[algo]['R']
            u_idx = int(models[algo]['user_idx'][user_id])
            rating_count = R.getrow(u_idx).nnz
            return rating_count < min_ratings, False
    return True, True

def get_popular_items(models: dict, n_recs: int = 10):
    """Get popular items as fallback"""
    for algo in ['user', 'item', 'model']:
        if algo in models:
            R = models[algo]['R']
            item_rev = models[algo]['item_rev']
            item_popularity = np.array(R.sum(axis=0)).ravel()
            top_idx = np.argsort(-item_popularity)[:n_recs]
            return [(item_rev[i], float(item_popularity[i])) for i in top_idx]
    return []

def predict_hybrid(user_id: str, models: dict, weights: dict = None):
    """Hybrid prediction with cold-start handling"""
    # Detect scenario
    scenario, n_ratings = detect_scenario(user_id, models, threshold=5)
    
    logger.log_info(f"[Hybrid] User: {user_id[:12]}... | Scenario: {scenario} | Ratings: {n_ratings}")
    
    # Get all predictions
    predictions = {}
    
    user_scores = predict_user(user_id, models)
    if user_scores is not None:
        predictions['user'] = user_scores
    
    item_scores = predict_item(user_id, models)
    if item_scores is not None:
        predictions['item'] = item_scores
    
    content_scores = predict_content(user_id, models)
    if content_scores is not None:
        predictions['content'] = content_scores
    
    model_scores = predict_model(user_id, models)
    if model_scores is not None:
        predictions['model'] = model_scores
    
    trending_scores = predict_trending(user_id, models)
    if trending_scores is not None:
        predictions['trending'] = trending_scores
    
    if not predictions:
        logger.log_info(f"[Hybrid] No predictions available")
        return None, "no_models"
    
    # Adaptive weights by scenario
    weight_configs = {
        'new-user': {
            'user': 0.0, 'item': 0.0, 'content': 0.0, 'model': 0.0, 'trending': 1.0
        },
        'cold-user': {
            'user': 0.1, 'item': 0.1, 'content': 0.3, 'model': 0.1, 'trending': 0.4
        },
        'warm-user': {
            'user': 0.25, 'item': 0.35, 'content': 0.20, 'model': 0.20, 'trending': 0.0
        }
    }
    
    available = list(predictions.keys())
    base_weights = weight_configs.get(scenario, weight_configs['warm-user'])
    
    # Filter to available models
    adaptive_weights = {m: w for m, w in base_weights.items() if m in available}
    
    # Adjust by activity level
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
    
    # Normalize
    total = sum(adaptive_weights.values())
    if total > 0:
        adaptive_weights = {k: v / total for k, v in adaptive_weights.items()}
    
    logger.log_info(f"[Hybrid] Models: {available}")
    logger.log_info(f"[Hybrid] Weights: {adaptive_weights}")
    
    # Combine predictions
    ref_model = available[0]
    n_items = len(predictions[ref_model])
    combined = np.zeros(n_items, dtype=np.float32)
    
    for model_name, scores in predictions.items():
        weight = adaptive_weights.get(model_name, 0.0)
        if weight > 0:
            combined += weight * scores
    
    # Build strategy string
    parts = [f"{m}({w*100:.0f}%)" for m, w in sorted(adaptive_weights.items(), key=lambda x: -x[1]) if w > 0.05]
    strategy = f"hybrid-{scenario}-" + "+".join(parts)
    
    logger.log_info(f"[Hybrid] Strategy: {strategy}")
    
    return combined, strategy

def predict_user(user_id: str, models: dict, k: int = 30):
    """User-based prediction"""
    if 'user' not in models or user_id not in models['user']['user_idx']:
        return None
    art = models['user']
    u = int(art['user_idx'][user_id])
    X = art['Rc'] if art.get('Rc') is not None else art['R']
    distances, indices = art['nn_model'].kneighbors(X.getrow(u), return_distance=True)
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

def predict_item(user_id: str, models: dict, k: int = 30):
    """Item-based prediction"""
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

def predict_content(user_id: str, models: dict, k: int = 30):
    """Content-based prediction"""
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

def predict_model(user_id: str, models: dict):
    """Model-based (SVD) prediction"""
    if 'model' not in models or user_id not in models['model']['user_idx']:
        return None
    art = models['model']
    u = int(art['user_idx'][user_id])
    return art['U'][u] @ art['V'].T

def get_recommendations(user_id: str, category: str, n_recs: int = 10):
    """Main recommendation function with hybrid logic"""
    models = load_hybrid_models(category)
    if not models:
        return [], "no_models"
    
    scores, strategy = predict_hybrid(user_id, models)
    
    if scores is None or strategy == "popular_only":
        popular = get_popular_items(models, n_recs)
        return popular, "popular"
    
    # Get reference R and item_rev
    for algo in ['user', 'item', 'content', 'model']:
        if algo in models:
            R = models[algo]['R']
            item_rev = models[algo]['item_rev']
            user_idx = models[algo]['user_idx']
            if user_id in user_idx:
                u = int(user_idx[user_id])
                rated = set(R.getrow(u).indices.tolist())
            else:
                rated = set()
            break
    
    # Filter out rated items
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

# ============= API ENDPOINTS =============

@app.route('/api/register', methods=['POST'])
def register():
    """User registration"""
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
    """User login"""
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
    """Get personalized recommendations using trained models"""
    username = get_jwt_identity()
    if username is None:
        username = 'guest'
    n_recs = request.args.get('n', default=10, type=int)
    
    try:
        # Load all trained models for hybrid recommendation
        recommendations, strategy = get_recommendations(username, category, n_recs)
        
        # Load metadata for product details and images
        meta_df = load_metadata(category)
        results = []
        
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
            
            # Enrich with metadata if available
            if meta_df is not None:
                meta_row = meta_df.filter(pl.col('parent_asin') == parent_asin)
                if len(meta_row) > 0:
                    row_dict = meta_row.to_dicts()[0]
                    
                    # Title
                    item_data['title'] = row_dict.get('title', 'Unknown Product')
                    
                    # Price
                    price_raw = row_dict.get('price', 'N/A')
                    if price_raw and price_raw != 'N/A':
                        item_data['price'] = price_raw
                    
                    # Rating
                    item_data['rating'] = float(row_dict.get('average_rating', 0) or 0)
                    item_data['rating_number'] = int(row_dict.get('rating_number', 0) or 0)
                    
                    # Images - handle list format
                    # Parse images - data format: [{'thumb': url, 'large': url, 'hi_res': url}, ...]
                    images = row_dict.get('images', [])
                    if isinstance(images, list) and len(images) > 0:
                        img_obj = images[0]
                        if isinstance(img_obj, dict):
                            # Ưu tiên hi_res > large > thumb
                            item_data['image_url'] = img_obj.get('hi_res') or img_obj.get('large') or img_obj.get('thumb')
                        elif isinstance(img_obj, str):
                            item_data['image_url'] = img_obj
                    else:
                        item_data['image_url'] = None
                    
                    # Features
                    features = row_dict.get('features', [])
                    if isinstance(features, list):
                        item_data['features'] = features[:5]  # Top 5 features
                    
                    # Description
                    desc = row_dict.get('description', [])
                    if isinstance(desc, list) and len(desc) > 0:
                        item_data['description'] = desc[0][:500]  # First 500 chars
            
            results.append(item_data)
        
        logger.log_info(f"[API] Returned {len(results)} recommendations for {username} in {category} using {strategy}")
        
        return jsonify({
            'recommendations': results,
            'strategy': strategy,
            'count': len(results),
            'user': username,
            'category': category
        }), 200
    
    except Exception as e:
        logger.log_exception(f"[API-Error] {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get available categories"""
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
    """Submit product rating"""
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
    """Get detailed product information"""
    category = request.args.get('category', 'Electronics')
    meta_df = load_metadata(category)
    
    if meta_df is not None:
        product = meta_df.filter(pl.col('parent_asin') == parent_asin)
        if len(product) > 0:
            product_dict = product.to_dicts()[0]
            # Process images
            images = product_dict.get('images', [])
            if isinstance(images, list):
                product_dict['images'] = [img.get('large', img.get('thumb', '')) for img in images if isinstance(img, dict)]
            return jsonify(product_dict), 200
    
    return jsonify({'error': 'Product not found'}), 404

@app.route('/api/placeholder-image', methods=['GET'])
def placeholder_image():
    """Serve placeholder image"""
    # Return a simple SVG placeholder
    svg = '''<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
        <rect width="200" height="200" fill="#f0f0f0"/>
        <text x="50%" y="50%" text-anchor="middle" fill="#999" font-size="16">No Image</text>
    </svg>'''
    return svg, 200, {'Content-Type': 'image/svg+xml'}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'models_loaded': len(MODELS_CACHE)}), 200

# ============= RUN SERVER =============

if __name__ == '__main__':
    print("Starting Amazon Product Recommendation System API...")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Data directory: {DATA_DIR}")
    app.run(debug=True, host='0.0.0.0', port=5000)