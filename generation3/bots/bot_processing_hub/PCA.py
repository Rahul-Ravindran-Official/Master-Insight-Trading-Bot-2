from sklearn.decomposition import PCA
from typing import List
import pandas as pd


class pcaUtility:

    @staticmethod
    def perform_pca(
            pca_features: List[str],
            data: pd.DataFrame,
            components: int
    ) -> pd.DataFrame:

        narrowed_down_data = data.loc[:, pca_features]
        pca = PCA(n_components=3)
        principal_components = pca.fit_transform(narrowed_down_data)

        pca_columns = []
        for i in range(components):
            pca_columns.append('principal component ' + str(i+1))

        principal_df = pd.DataFrame(
            data=principal_components,
            columns=pca_columns
        )

        return principal_df
