===============================================================================
      	 AMAZON PRODUCT RECOMMENDATION SYSTEM - IE7275 - Group 6
                    	    Quick Reference
===============================================================================


 PROJECT OVERVIEW
-------------------------------------------------------------------------------
Goal: Build production-ready recommendation system with cold-start handling
More achivement: Handle full cold-start (users, items); real-time recommend by user'rating
Dataset: Amazon 2023 Reviews (3 categories)
Algorithms: 6 (User-CF, Item-CF, Content, SVD, Trending, Hybrid)
Deployment: Full-stack web app (Flask + React)

-------------------------------------------------------------------------------
 PROJECT STRUCTURE
-------------------------------------------------------------------------------

Amazon-Product-Recommendation-System/
|
+-- data/
|   +-- raw/              <- Original CSV.gz files
|   +-- processed/        <- Parquet files (train/valid/test)
|
+-- notebooks/            <- Jupyter notebooks (ALL PRE-EXECUTED)
|   +-- data_collection.ipynb
|   +-- exploratory_data.ipynb
|   +-- collaborative_filtering/
|   |   +-- user_based.ipynb
|   |   +-- item_based.ipynb
|   |   +-- content_based.ipynb
|   |   +-- model_based.ipynb
|   |   +-- trending_based.ipynb
|   +-- hybrid_system.ipynb
|
+-- models/               <- Pre-trained models + results
|   +-- user/             <- User-CF models
|   +-- item/             <- Item-CF models
|   +-- content/          <- Content-based models
|   +-- model/            <- SVD models
|   +-- trending/         <- Trending models
|   +-- hybrid/           <- Hybrid ensemble
|   +-- comparison_all_models.csv    [KEY FILE]
|   +-- comparison_all_metrics.png   [KEY FILE]
|   +-- comparison_per_category.png  [KEY FILE]
|
+-- utilities/            <- Helper functions
|   +-- configuration.py  [UPDATE ROOT_DIR HERE]
|   +-- logger.py
|   +-- ...
|
+-- frontend/             <- React web application
|   +-- src/
|   +-- package.json
|   +-- vite.config.js
|
+-- app.py                <- Flask API backend
+-- requirements.txt      <- Python dependencies
|
+-- Report/
|   +-- IE7275_Group Project1_Fall2025_Group 6.pdf  [MAIN REPORT]
|
+-- README.md  [START HERE - 5-min guide]
 
TIME ESTIMATES
==============
    - OPTION 1: Review report and models & metrics:    15 minutes
    - OPTION 2: Run web demo:                          5 minutes
    - OPTION 3: Full:                     1-2 hours (Depend on sample type and machine)
RECOMMENDATION: Start with notebooks review + web demo (20 min total)

Known Issues (Acknowledged in Report)
=====================================
    1. Perfect Recall Artifact: Small test value set causes 1.0 recall (not real-world)
    2. Hybrid Underperforms: Needs weight optimization
    3. Sample Size: 50K samples vs millions (speed vs accuracy tradeoff)

Key Innovations
==============
    1. Real-time updates: Ratings instantly refresh recommendations  
    2. Adaptive hybrid: Weights change based on user scenario 

OPTION 1: FASTEST
=================
Just review report in ./Report/ and pre-computed results:                                        
    1. Open models/comparison_all_models.csv                        
    2. Open models/*.png visualizations             
    3. Scroll through notebooks/ (outputs already visible)
    !!! NO CODE EXECUTION NEEDED  

OPTION 2: LIVE DEMO
===================
First:
    1. cd Amazon-Product-Recommendation-System and run ./notebooks/data_collection.ipynb (This download metadata for enrich UI)
    2. python -m venv venv && source venv/bin/activate  # One-time setup
    3. pip install -r requirements.txt                  # One-time install

Next:
    1. Terminal 1: python app.py                                                
    2. Terminal 2: cd frontend && npm run dev                                  
    3. Browser: http://localhost:5173                                           
    !!! Test real-time recommendations as workfollow:

User Create and Login                                                                  
      ↓                                                                       
Detect Scenario ──→ New (0 ratings)    → Trending only (100%)             
      ↓               Cold (1-4 ratings)  → Trending(40%) + Content(30%)     
      ↓               Warm (5-20)         → Full hybrid with CF              
      ↓               Active (>20)        → CF-focused (less trending)       
      ↓                                                                       
Load Models ──→ user/item/content/model/trending                          
      ↓                                                                      
Predict Scores ──→ Weighted combination per scenario                      
	 ↓                                                                     
Rank Items ──→ Exclude rated items (train + rating_history)              
     ↓                                                                       
Return Top-K ──→ Enrich with metadata (title, price, images)             
     ↓                                                                       
Display UI ──→ Show scenario badge + algorithm strategy                   
    ↓                                                                       
User Rates ──→ Update rating_history → Refresh recommendations        
     ↓                                                                       
Real-time Update (NO model retraining needed!

OPTION 3: FULL REPRODUCTION
===========================
First:
    1. Delete all files in /data/, /models/
    2. Edit ROOT_DIR in utilities/configuration.py to local project's directory on your machine

Next, do as workfollow:
    1. data_collection.ipynb --> check ./data/ + ./logs
    2. explorary_data.ipynb --> follow cell output + ./logs
    3. run separately all files in ./notebooks/ --> check ./models/ and follow cell output 
    4. run hybrid_system.ipynb--> check ./models/ and follow cell output
    5. open terminal 1: python app.py
    6. open terminal 2: cd frontend && npm run dev

Common troubleshooting
======================
Issue: Module not found
Fix: pip install -r requirements.txt

Issue: File not found (data/models)
Fix: Update ROOT_DIR in utilities/configuration.py

Issue: Port 5000 already in use
Fix: lsof -ti:5000 | xargs kill  (Mac/Linux)
netstat -ano | findstr :5000  (Windows)

Issue: Out of memory
Fix: Reduce sample size or close other applications

Issue: npm dependencies fail
Fix: rm -rf node_modules package-lock.json && npm install








===============================================================================================
PART B: DETAIL DEPLOYMENT
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
+ Data Pipeline
    Collection from Amazon 2023
    5 categories
    Activity-based filtering
    Train/Valid/Test splits

+ 5 Recommendation Models
    User-Based CF
    Item-Based CF
    Content-Based
    Model-Based (SVD/ALS)
    Hybrid Ensemble

+ Evaluation & Tuning
    NDCG@10, Recall@10, MAP@10
    Hyperparameter optimization
    Validation framework

+ Backend API
    Flask REST API
    JWT authentication
    Model serving

+ Frontend UI
    React interface
    User login/register
    Product display

+ Project Structure
    Clean code organization
    Modular design
    Version control ready
