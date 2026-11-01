from ..utils.logging import get_logger


class EarlyStopping:
    """Monitors a metric that should IMPROVE (maximize by default)."""

    def __init__(
        self,
        patience: int = 20,
        delta: float = 1e-3,
        mode: str = "max",
        verbose: bool = True,
    ):
        assert mode in ("max", "min")
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best = None
        self.early_stop = False

    def _improved(self, score: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return score > self.best + self.delta
        return score < self.best - self.delta

    def __call__(self, score: float) -> None:
        if self._improved(score):
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                get_logger().info(
                    f"  EarlyStopping {self.counter}/{self.patience} "
                    f"(best={self.best:.4f})"
                )
            if self.counter >= self.patience:
                self.early_stop = True