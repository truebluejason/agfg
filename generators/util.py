from pandas import DataFrame, concat

NUMERICS = ['float16', 'float32', 'float64']


class StandardScaler:
    """
    Returns a new DataFrame where z-score scaling is applied to
    all numeric features
    """

    def __init__(self) -> None:
        self.means = None
        self.stds = None

    def fit_transform(self, df: DataFrame):
        numeric_cols = df.select_dtypes(include=NUMERICS)
        self.means = numeric_cols.mean(axis=0)
        self.stds = numeric_cols.std(axis=0)
        return self.transform(df)

    def transform(self, df: DataFrame) -> DataFrame:
        non_numeric_cols = df.select_dtypes(exclude=NUMERICS)
        numeric_cols = df.select_dtypes(include=NUMERICS)
        standardized = (numeric_cols - self.means) / (self.stds + 1e-6)
        return concat([non_numeric_cols, standardized], axis=1)
