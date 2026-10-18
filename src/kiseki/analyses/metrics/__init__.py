import msgspec

TITLE = "Accuracy and per-class metrics"
SUMMARY = "Accuracy, loss, macro F1, per-class precision/recall/F1 and confusion matrices."


class Params(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    pass
