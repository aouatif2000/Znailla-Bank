import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def preprocess_data(df, encoder=None):
    df = df.copy()
    if 'duration' in df.columns:
        df.drop('duration', axis=1, inplace=True)
    y = df['y'].map({'yes': 1, 'no': 0}) if 'y' in df.columns else None
    X = df.drop(columns=['y'], errors='ignore')
    cat_cols = X.select_dtypes(include='object').columns.tolist()
    num_cols = X.select_dtypes(exclude='object').columns.tolist()
    if encoder is None:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_cat = encoder.fit_transform(X[cat_cols])
    else:
        X_cat = encoder.transform(X[cat_cols])
    X_cat_df = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(cat_cols), index=X.index)
    X_final = pd.concat([X[num_cols], X_cat_df], axis=1)
    return X_final, y, encoder
