from pathlib import Path
import json

class Configurations:
    
    # Paths
    ROOT_DIR = '/Users/kevin/Documents/GitHub/Python/VESKL/Personal/NEU/NEU/NEU_7275/Prj/Amazon-Product-Recommendation-System'
    LOG_PATH = ROOT_DIR + '/logs/'
    DATA_RAW_PATH = ROOT_DIR + '/data/raw/'
    DATA_PROCESSED_PATH = ROOT_DIR + '/data/processed/'   
    MODELS_PATH = ROOT_DIR + '/models/'
    
    # Dataset
    CATEGORIES = ["Electronics","Beauty_and_Personal_Care", "Sports_and_Outdoors"]#, "Baby_Products", "Cell_Phones_and_Accessories"]
    CORES = ["5core"]
    SPLITS = ["train", "valid", "test"]
    SPLITS_EDA = ["train"]
    COLUMNS = ['user_id', 'parent_asin', 'rating', 'timestamp']
    BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark"
    META_BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_{category}.jsonl.gz"
    
    # Sampling
    SAMPLE_SIZES = {'big':68000} #{'small': 2000, 'medium': 20000, 'large':50000, 'big':50000, 'full': None}
    ITEM_MULTI = 79
    DEV_SAMPLE_SIZE = "big"
    
    @classmethod
    def get_train_file_path(cls, category: str, size: str = None):
        size = size or cls.DEFAULT_SAMPLE_SIZE
        safe_cat = category.replace('/', '-')
        
        if size == 'full':
            filename = f"{safe_cat}.5core.train.parquet"
        else:
            filename = f"{safe_cat}.5core.train.{size}.parquet"
        
        return Path(cls.DATA_PROCESSED_PATH) / filename
    
    @classmethod
    def get_valid_file_path(cls, category: str, size: str = None):
        size = size or cls.DEFAULT_SAMPLE_SIZE
        safe_cat = category.replace('/', '-')
        
        if size == 'full':
            filename = f"{safe_cat}.5core.valid.parquet"
        else:
            filename = f"{safe_cat}.5core.valid.{size}.parquet"
        
        return Path(cls.DATA_PROCESSED_PATH) / filename
    
    @classmethod
    def get_test_file_path(cls, category: str, size: str = None):
        size = size or cls.DEFAULT_SAMPLE_SIZE
        safe_cat = category.replace('/', '-')
        
        if size == 'full':
            filename = f"{safe_cat}.5core.test.parquet"
        else:
            filename = f"{safe_cat}.5core.test.{size}.parquet"
        
        return Path(cls.DATA_PROCESSED_PATH) / filename
    
    @classmethod
    def get_meta_file_path(cls, category: str) -> Path:
        safe_cat = category.replace('/', '-')
        return Path(cls.DATA_PROCESSED_PATH) / f"{safe_cat}.meta.parquet"
    
    # Trending params
    TOP_N = 10
    RECENT_DAYS = 90
    PRIOR_COUNT = 10.0
    W_BAYES = 0.45
    W_RECENT = 0.40
    W_POP = 0.15
    CHUNKSIZE = 2_000_000
    DTYPE_RATING = "float32"
    DTYPE_ASIN = "category"
    
    # Evaluation
    EVAL_SAMPLES_TUNING = {'small': 1000, 'medium': 10000, 'large':100000, 'full': None}
    EVAL_SAMPLES_FINAL = {'small': 2000, 'medium': 20000, 'large':200000, 'full': None}
    EVAL_SAMPLES = 10000  # Legacy
    
    @classmethod
    def get_eval_samples_tuning(cls, sample_size: str = None) -> int:
        if sample_size is None:
            sample_size = cls.DEV_SAMPLE_SIZE
        return cls.EVAL_SAMPLES_TUNING.get(sample_size, 3000)
    
    @classmethod
    def get_eval_samples_final(cls, sample_size: str = None) -> int:
        if sample_size is None:
            sample_size = cls.DEV_SAMPLE_SIZE
        return cls.EVAL_SAMPLES_FINAL.get(sample_size, 1000)

    
    # ========================================================================
    # HYPERPARAMETER RANGES - Different per algorithm!
    # ========================================================================
    
    K_VALUES_COARSE = [10, 20, 50, 100]  # User-based k_neighbors
    K_VALUES_ITEM = [10, 20, 50]     # Item-based top_k_similar
    K_VALUES_CONTENT = [10] # Content-based top_k_similar
    FACTORS_VALUES = [10, 20, 30, 50, 75]  # Model-based n_factors
    
    # ========================================================================
    # USER-BASED CF - k_neighbors (number of similar users)
    # ========================================================================
    
    @staticmethod
    def save_best_k(category: str, k: int):
        """Save best k_neighbors for user-based"""
        config_file = Path(Configurations.MODELS_PATH) / 'user' / 'best_k_config.json'
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        config = {}
        if config_file.exists():
            config = json.loads(config_file.read_text())
        
        config[category] = k
        config_file.write_text(json.dumps(config, indent=2))
    
    @staticmethod
    def load_best_k(category: str) -> int:
        """Load best k_neighbors for user-based (default: 30)"""
        config_file = Path(Configurations.MODELS_PATH) / 'user' / 'best_k_config.json'
        
        if not config_file.exists():
            return 30
        
        config = json.loads(config_file.read_text())
        return config.get(category, 30)
    
    @staticmethod
    def has_tuning_results_user(category: str) -> bool:
        """Check if user-based tuning done"""
        tuning_file = Path(Configurations.MODELS_PATH) / 'user' / f'tuning_{category}.csv'
        return tuning_file.exists()
    
    # ========================================================================
    # ITEM-BASED CF - top_k_similar (co-rating based similarity)
    # ========================================================================
    
    @staticmethod
    def save_best_k_item(category: str, k: int):
        """Save best top_k_similar for item-based"""
        config_file = Path(Configurations.MODELS_PATH) / 'item' / 'best_k_config.json'
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        config = {}
        if config_file.exists():
            config = json.loads(config_file.read_text())
        
        config[category] = k
        config_file.write_text(json.dumps(config, indent=2))
    
    @staticmethod
    def load_best_k_item(category: str) -> int:
        """Load best top_k_similar for item-based (default: 30)"""
        config_file = Path(Configurations.MODELS_PATH) / 'item' / 'best_k_config.json'
        
        if not config_file.exists():
            return 30
        
        config = json.loads(config_file.read_text())
        return config.get(category, 30)
    
    @staticmethod
    def has_tuning_results_item(category: str) -> bool:
        """Check if item-based tuning done"""
        tuning_file = Path(Configurations.MODELS_PATH) / 'item' / f'tuning_{category}.csv'
        return tuning_file.exists()
    
    # ========================================================================
    # CONTENT-BASED CF - top_k_similar (text/metadata similarity)
    # ========================================================================
    
    @staticmethod
    def save_best_k_content(category: str, k: int):
        """Save best top_k_similar for content-based"""
        config_file = Path(Configurations.MODELS_PATH) / 'content' / 'best_k_config.json'
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        config = {}
        if config_file.exists():
            config = json.loads(config_file.read_text())
        
        config[category] = k
        config_file.write_text(json.dumps(config, indent=2))
    
    @staticmethod
    def load_best_k_content(category: str) -> int:
        """Load best top_k_similar for content-based (default: 30)"""
        config_file = Path(Configurations.MODELS_PATH) / 'content' / 'best_k_config.json'
        
        if not config_file.exists():
            return 30
        
        config = json.loads(config_file.read_text())
        return config.get(category, 30)
    
    @staticmethod
    def has_tuning_results_content(category: str) -> bool:
        """Check if content-based tuning done"""
        tuning_file = Path(Configurations.MODELS_PATH) / 'content' / f'tuning_{category}.csv'
        return tuning_file.exists()
    
    # ========================================================================
    # MODEL-BASED CF - n_factors (latent dimensions, NOT K!)
    # ========================================================================
    
    @staticmethod
    def save_best_factors(category: str, n_factors: int):
        """Save best n_factors for model-based"""
        config_file = Path(Configurations.MODELS_PATH) / 'model' / 'best_factors_config.json'
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        config = {}
        if config_file.exists():
            config = json.loads(config_file.read_text())
        
        config[category] = n_factors
        config_file.write_text(json.dumps(config, indent=2))
    
    @staticmethod
    def load_best_factors(category: str) -> int:
        """Load best n_factors for model-based (default: 50)"""
        config_file = Path(Configurations.MODELS_PATH) / 'model' / 'best_factors_config.json'
        
        if not config_file.exists():
            return 50
        
        config = json.loads(config_file.read_text())
        return config.get(category, 50)
    
    @staticmethod
    def has_tuning_results_model(category: str) -> bool:
        """Check if model-based tuning done"""
        tuning_file = Path(Configurations.MODELS_PATH) / 'model' / f'tuning_{category}.csv'
        return tuning_file.exists()
    
    # ========================================================================
    # TRENDING MODEL
    # ========================================================================
    def has_tuning_results_trending(category: str) -> bool:
        model_dir = Path(Configurations.MODELS_PATH) / 'trending' / category
        return (model_dir / "R.npz").exists()