I. INSTALLATION
- Setup Environment
bash# Navigate to project
cd Amazon-Product-Recommendation-System

# Create virtual environment
python -m venv venv

# Activate
source venv/bin/activate      # Mac/Linux

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt  # Maybe need more, just please fix in running


II. MANUAL RUN
1. Download dataset (5core and meta), build sub-dataset/metadata and save to parquet file format.
INPUT                 PROCESS                OUTPUT
┌────────────┐       ┌────────────┐        ┌────────────┐
│ Amazon     │       │ Filter     │        │ Train      │
│ Raw Data   │──────▶│ 5-core     │───────▶│ Valid      │
│            │       │ Clean      │        │ Test       │
│ CSV.gz     │       │ Split      │        │            │
└────────────┘       └────────────┘        └────────────┘
                                            *.parquet files
- Edit in /utilites/conofiguration.py
ROOT_DIR = '/Users/kevin/Documents/GitHub/Python/VESKL/Personal/NEU/NEU/NEU_7275/Prj/Amazon-Product-Recommendation-System' --> Change to your working directory.
CATEGORIES = ["Electronics","Beauty_and_Personal_Care", "Sports_and_Outdoors"] --> Just choose 2-3, more longtime
- Run all data_collection.ipynb --> Check in ./data/raw_data/ and ./data/preprocessed_data/

2. Visualization
- Run all exploratory_analysis.ipynb --> Just view in cell output.

3. Training model, unit test, evaluation
- Run all *_based.ipynb in /notebooks/ and /notebooks/collaborativve_filtering/ --> Check in ./models/
- Manually remove all in /models/ if trying rerun

4. Check logs
- Check logs in ./logs/

5. Try to run with website
API Request Flow
USER                 FRONTEND              BACKEND              MODELS
 │                      │                     │                    │
 │  Click "Get Recs"    │                     │                    │
 ├─────────────────────▶│                     │                    │
 │                      │  POST /api/login    │                    │
 │                      ├────────────────────▶│                    │
 │                      │                     │ Verify password    │
 │                      │  Return JWT token   │                    │
 │                      │◀────────────────────┤                    │
 │                      │                     │                    │
 │                      │  GET /recommend     │                    │
 │                      ├────────────────────▶│                    │
 │                      │  + JWT token        │ Load models        │
 │                      │  + category         │◀───────────────────┤
 │                      │                     │                    │
 │                      │                     │ Get predictions    │
 │                      │                     ├───────────────────▶│
 │                      │                     │ Return scores      │
 │                      │                     │◀───────────────────┤
 │                      │                     │                    │
 │                      │  Return JSON        │ Rank items         │
 │                      │  [items...]         │                    │
 │  Show products       │◀────────────────────┤                    │
 │◀─────────────────────┤                     │                    │
 │                      │                     │                    │
- Open two termial and run 2 commands in current working directory:
    bash# python app.py
    bash# cd frontend && npm run dev
- Go to http://localhost:5173
- No login, just show trending producsts (Now UI only specially for some categories)
- Create username/password for login
+ Newuser = Cold-start user, no history of purchase --> Hybrid-based recommendation system
+ Olduser = Warm-start user, have history of purchase

III. Current Status
- A complete offline recommendation system with:
Data collection & preprocessing/Feature engineering
Exploratory analysis
Multiple algorithms (CF:item/user/model-based, content-based, hybrid)
Full data pipeline
Evaluation framework
Web interface
API backend
- 6 Major Components Done:
+ Data Pipeline (95%)
    Collection from Amazon 2023
    5 categories
    Activity-based filtering
    Train/Valid/Test splits

+ 5 Recommendation Models (90%)
    User-Based CF
    Item-Based CF
    Content-Based
    Model-Based (SVD/ALS)
    Hybrid Ensemble

+ Evaluation & Tuning (85%)
    NDCG@10, Recall@10, MAP@10
    Hyperparameter optimization
    Validation framework

+ Backend API (80%)
    Flask REST API
    JWT authentication
    Model serving

+ Frontend UI (75%)
    React interface
    User login/register
    Product display

+ Project Structure (95%)
    Clean code organization
    Modular design
    Version control ready

--> NEXT:
- Streaming data ingestion the real-time behavior of user interaction with the website: Collection, online model updates, Instant recommendations
- Improve performance (speed, evaluation metrics, etc.), maybe new algorithms, hyperparameter tuning, etc.
- Finalize all project: src, .....
- Prepare detail demo and visualization, evaluation report (special of hybrid).
