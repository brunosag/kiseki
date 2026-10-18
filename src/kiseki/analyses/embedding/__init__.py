from typing import Annotated

import msgspec
from msgspec import Meta

TITLE = "Embeddings of the penultimate layer"
SUMMARY = (
    "PCA and t-SNE maps of each model's penultimate activations, coloured by class; "
    "with kNN accuracy and trustworthiness as numbers to compare."
)


class Params(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    samples: Annotated[
        int, Meta(ge=50, description="Images embedded per model (class-balanced)")
    ] = 5000
    perplexity: Annotated[float, Meta(ge=2, le=100)] = 30.0
    iterations: Annotated[int, Meta(ge=250, le=5000)] = 1000
    pca_components: Annotated[int, Meta(ge=2, le=200, description="PCA before t-SNE")] = 50
    neighbors: Annotated[
        int, Meta(ge=1, le=100, description="k for kNN accuracy and trustworthiness")
    ] = 10
