from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    qwen_model: str = "models/Qwen3-VL-8B-Instruct"
    sam_checkpoint: str = "models/sam3/sam3.pt"
    # Uniform sampling rounds, each with a different temporal offset.
    frames_per_round: int = 64
    iterations: int = 4
    # Equals the original frame size, so no call downscales images.
    max_pixels: int = 1920 * 1440
    sam_confidence: float = 0.5
    overlay_alpha: float = 0.7
    agreement_threshold: float = 0.7
    visibility_tolerance: float = 0.25
    max_new_tokens: int = 512
    seed: int = 0
    # Frames kept per selection call.
    max_selected_frames: int = 4
    # These implementation details are unspecified in the paper.
    upright: bool = True
    attention: str = "sdpa"
    linear_patch_embed: bool = True
    max_videos: int = 0  # 0 = all benchmark videos

    def __post_init__(self):
        if self.frames_per_round < 2 or self.iterations < 1:
            raise ValueError("Need at least two frames per round and one iteration")
        if self.max_videos < 0:
            raise ValueError("Video limit must be nonnegative")
        if self.max_selected_frames < 1:
            raise ValueError("Frame limit must be positive")
        if min(self.max_pixels, self.max_new_tokens) <= 0:
            raise ValueError("Pixel and token budgets must be positive")
        for x in (self.sam_confidence, self.overlay_alpha, self.agreement_threshold):
            if not 0 <= x <= 1:
                raise ValueError("Thresholds must lie in [0, 1]")
