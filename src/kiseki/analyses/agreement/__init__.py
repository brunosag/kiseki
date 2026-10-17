import msgspec

TITLE = "Agreement between models"
SUMMARY = (
    "Which test images each model gets right, how often their predictions agree, "
    "and McNemar's exact test of their error rates."
)


class Params(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    pass
