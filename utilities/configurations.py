class Configurations():
    # [Workspace]
    ROOT_DIR = '/Users/kevin/Documents/GitHub/Python/VESKL/Personal/NEU/NEU/NEU_7275/Prj/Prj_1/APRS_7275_G6/Amazon-Product-Recommendation-System'#os.getcwd()
    LOG_PATH = ROOT_DIR + '/logs/'
    DATA_RAW_PATH = ROOT_DIR + '/data/raw/'
    DATA_PROCESSED_PATH = ROOT_DIR + '/data/processed/'   
    MODELS_PATH = ROOT_DIR + '/models/'
    # FIGURES_PATH = ROOT_DIR + '/figures/' 

    # [Resource Dataset Amazon Reviews dataset, collected in 2023 by McAuley Lab]
    CATEGORIES = ["Electronics", "Beauty_and_Personal_Care"]
    CORES = ["0core", "5core"]
    SPLITS = ["train", "valid", "test"]
    SPLITS_EDA = ["train"]  # for EDA, only need train splits
    COLUMNS = ['user_id', 'parent_asin', 'rating', 'timestamp', 'history']
    BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark"
    META_BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_{category}.jsonl.gz"


    # [Trending Parameters]
    TOP_N          = 10     # top trending items per category
    RECENT_DAYS    = 90     # how far back to consider "recent" activity
    PRIOR_COUNT    = 10.0   # Bayesian smoothing prior strength
    W_BAYES        = 0.45   # weight: quality (Bayesian average)
    W_RECENT       = 0.40   # weight: freshness (recent share)
    W_POP          = 0.15   # weight: popularity (log count)
    CHUNKSIZE      = 2_000_000  # rows per chunk for streaming large CSVs

    # DType hints to reduce memory (timestamp will be coerced safely)
    DTYPE_RATING   = "float32"
    DTYPE_ASIN     = "category"  # enforced post-read