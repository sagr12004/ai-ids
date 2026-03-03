import pandas as pd
import os
from sklearn.model_selection import train_test_split

columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
    "srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"
]

def load_data():
    train = pd.read_csv("data/KDDTrain+.csv", header=None, names=columns)
    test = pd.read_csv("data/KDDTest+.csv", header=None, names=columns)
    train.drop("difficulty", axis=1, inplace=True)
    test.drop("difficulty", axis=1, inplace=True)
    train["label"] = train["label"].apply(lambda x: "normal" if x == "normal" else "attack")
    test["label"] = test["label"].apply(lambda x: "normal" if x == "normal" else "attack")
    os.makedirs("artifacts", exist_ok=True)
    train.to_csv("artifacts/train.csv", index=False)
    test.to_csv("artifacts/test.csv", index=False)
    print("Data ingestion complete.")
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    return train, test

if __name__ == "__main__":
    load_data()
