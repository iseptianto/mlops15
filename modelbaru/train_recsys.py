# modelbaru/train_recsys.py
import os, argparse, pickle, warnings
import pandas as pd
import numpy as np
import mlflow, mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

warnings.filterwarnings("ignore")

class RecsysModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import pickle, os
        self.popular = pickle.load(open(context.artifacts["popular"], "rb"))
        self.user_seen = pickle.load(open(context.artifacts["user_seen"], "rb"))

    def predict(self, context, model_input: pd.DataFrame):
        """
        Predict recommendations for users
        Input: DataFrame with 'user_id' column
        Output: List of recommendations for each user
        """
        out = []
        for uid in model_input["user_id"].astype(str).tolist():
            seen = self.user_seen.get(uid, set())
            recs = [{"place": str(i), "score": 1.0} for i in self.popular if str(i) not in seen][:10]
            out.append(recs)
        return out

def create_dummy_data(output_path):
    """Create dummy tourism rating data for testing"""
    np.random.seed(42)
    n_users, n_places = 100, 50
    
    # Generate random ratings
    data = []
    for user_id in range(1, n_users + 1):
        n_ratings = np.random.randint(3, 15)  # Each user rates 3-15 places
        places = np.random.choice(range(1, n_places + 1), n_ratings, replace=False)
        for place_id in places:
            rating = np.random.choice([3, 4, 5], p=[0.2, 0.3, 0.5])  # Bias toward higher ratings
            data.append({
                'user_id': user_id,
                'place_id': place_id,
                'rating': rating
            })
    
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Created dummy dataset with {len(df)} ratings: {output_path}")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings_csv", default="/app/data/dataset.csv")
    ap.add_argument("--user_col", default="user_id")
    ap.add_argument("--item_col", default="place_id")
    ap.add_argument("--rating_col", default="rating")
    ap.add_argument("--registered_model", default=os.getenv("MLFLOW_REGISTERED_MODEL","tourism-recommender-model"))
    args = ap.parse_args()

    # Setup MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","http://mlflow_server:5001"))
    experiment_name = os.getenv("MLFLOW_EXPERIMENT", "tourism-recommendation")
    
    try:
        mlflow.create_experiment(experiment_name)
    except:
        pass
    
    mlflow.set_experiment(experiment_name)

    # Load or create data
    if not os.path.exists(args.ratings_csv):
        print(f"Dataset not found at {args.ratings_csv}, creating dummy data...")
        df = create_dummy_data(args.ratings_csv)
    else:
        df = pd.read_csv(args.ratings_csv)
    
    # Normalize column names
    df = df.rename(columns={
        "User_Id": "user_id", 
        "Place_Id": "place_id", 
        "Place_Ratings": "rating"
    })
    
    # Clean data
    required_cols = [args.user_col, args.item_col, args.rating_col]
    df = df[required_cols].dropna()
    df[args.user_col] = df[args.user_col].astype(str)
    df[args.item_col] = df[args.item_col].astype(str)

    print(f"Training with {len(df)} ratings, {df[args.user_col].nunique()} users, {df[args.item_col].nunique()} items")

    # Calculate popularity (by mean rating, then count)
    agg = (df.groupby(args.item_col)
             .agg(mean_rating=(args.rating_col,"mean"),
                  cnt=(args.rating_col,"count"))
             .reset_index())
    agg = agg.sort_values(["mean_rating","cnt"], ascending=[False, False])
    popular_items = agg[args.item_col].tolist()

    # User interaction history
    user_seen = (df.groupby(args.user_col)[args.item_col]
                   .apply(lambda s: set(s.astype(str).tolist()))
                   .to_dict())

    # Save artifacts
    os.makedirs("artifacts", exist_ok=True)
    pickle.dump(popular_items, open("artifacts/popular.pkl","wb"))
    pickle.dump(user_seen, open("artifacts/user_seen.pkl","wb"))

    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params({
            "n_users": df[args.user_col].nunique(),
            "n_items": df[args.item_col].nunique(),
            "n_ratings": len(df),
            "model_type": "popularity_based",
        })
        
        # Log metrics
        mlflow.log_metrics({
            "avg_rating": df[args.rating_col].mean(),
            "rating_std": df[args.rating_col].std(),
            "sparsity": 1 - (len(df) / (df[args.user_col].nunique() * df[args.item_col].nunique()))
        })

        # Create input example for signature
        input_example = pd.DataFrame({"user_id": df[args.user_col].head(3).tolist()})
        
        # Log model
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=RecsysModel(),
            artifacts={
                "popular": "artifacts/popular.pkl",
                "user_seen": "artifacts/user_seen.pkl"
            },
            registered_model_name=args.registered_model,
            signature=infer_signature(input_example, None),
            input_example=input_example
        )

        # Set production alias
        client = MlflowClient()
        models = client.search_model_versions(f"name='{args.registered_model}'")
        latest_version = max([int(m.version) for m in models if m.run_id == run.info.run_id])
        
        client.set_registered_model_alias(args.registered_model, "production", str(latest_version))
        print(f"✅ Set alias 'production' -> {args.registered_model} v{latest_version}")
        print(f"✅ Run ID: {run.info.run_id}")

if __name__ == "__main__":
    main()