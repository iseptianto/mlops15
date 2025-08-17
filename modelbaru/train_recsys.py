# modelbaru/train_recsys.py
import os, argparse, pickle
import pandas as pd
import mlflow, mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

# misal df_train adalah DataFrame input (user_id, item_id, rating)
input_example = df.head(5)[["user_id", "item_id"]]   # atau kolom fitur yang dipakai predict
y_example = model.predict(input_example)             # sesuaikan API modelmu

signature = infer_signature(input_example, y_example)

mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    registered_model_name=os.getenv("MLFLOW_REGISTERED_MODEL", "tourism-recommender-model"),
    signature=signature,
    input_example=input_example,
)
s
class RecsysModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import pickle, os
        self.popular = pickle.load(open(os.path.join(context.artifacts["popular"]), "rb"))
        self.user_seen = pickle.load(open(os.path.join(context.artifacts["user_seen"]), "rb"))

    def predict(self, context, model_input: pd.DataFrame):
        out = []
        for uid in model_input["user_id"].astype(str).tolist():
            seen = self.user_seen.get(uid, set())
            recs = [int(i) for i in self.popular if i not in seen][:10]
            out.append(recs)
        return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings_csv", default="/app/data/tourism_rating.csv")
    ap.add_argument("--user_col", default="user_id")
    ap.add_argument("--item_col", default="place_id")
    ap.add_argument("--rating_col", default="rating")
    ap.add_argument("--registered_model", default=os.getenv("MLFLOW_REGISTERED_MODEL","tourism-recommender-model"))
    args = ap.parse_args()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","http://mlflow_server:5001"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT","modelbaru"))

    df = pd.read_csv(args.ratings_csv)
    df = df.rename(columns={
        "User_Id":"user_id", "Place_Id":"place_id", "Place_Ratings":"rating"
    })
    df = df[[args.user_col, args.item_col, args.rating_col]].dropna()
    df[args.user_col] = df[args.user_col].astype(str)
    df[args.item_col] = df[args.item_col].astype(str)

    # Popular items by mean rating, break ties by count
    agg = (df.groupby(args.item_col)
             .agg(mean_rating=(args.rating_col,"mean"),
                  cnt=(args.rating_col,"count"))
             .reset_index())
    agg = agg.sort_values(["mean_rating","cnt"], ascending=[False, False])
    popular_items = agg[args.item_col].tolist()

    # User -> set(items already seen)
    user_seen = (df.groupby(args.user_col)[args.item_col]
                   .apply(lambda s: set(s.tolist()))
                   .to_dict())

    os.makedirs("artifacts", exist_ok=True)
    pickle.dump(popular_items, open("artifacts/popular.pkl","wb"))
    pickle.dump(user_seen, open("artifacts/user_seen.pkl","wb"))

    with mlflow.start_run() as run:
        mlflow.log_artifact("artifacts/popular.pkl")
        mlflow.log_artifact("artifacts/user_seen.pkl")

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=RecsysModel(),
            artifacts={"popular":"artifacts/popular.pkl",
                       "user_seen":"artifacts/user_seen.pkl"},
            registered_model_name=args.registered_model,
        )

        # set alias @production ke versi yang barusan
        client = MlflowClient()
        mv = next(int(m.version) for m in client.search_model_versions(f"name='{args.registered_model}'")
                  if m.run_id == run.info.run_id)
        client.set_registered_model_alias(args.registered_model, "production", mv)
        print(f"Set alias 'production' -> {args.registered_model} v{mv}")

if __name__ == "__main__":
    main()
