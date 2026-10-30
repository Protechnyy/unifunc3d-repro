from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    qwen_model: str = "models/Qwen3-VL-8B-Instruct"
    sam_checkpoint: str = "models/sam3/sam3.pt"
    coarse_frames: int = 64
    iterations: int = 4
    coarse_width: int = 512
    coarse_height: int = 384
    fine_max_pixels: int = 1920 * 1440
    sam_confidence: float = 0.5
    overlay_alpha: float = 0.7
    agreement_threshold: float = 0.7
    visibility_tolerance: float = 0.25
    max_new_tokens: int = 192
    seed: int = 0
    # These implementation details are unspecified in the paper.
    upright: bool = True
    attention: str = "sdpa"
    linear_patch_embed: bool = True
    part_aware_prompt: bool = True
    max_fine_frames: int = 0  # 0 = every frame in every selected window
    max_videos: int = 0  # 0 = all benchmark videos

    def __post_init__(self):
        if self.coarse_frames < 2 or self.iterations < 1:
            raise ValueError("Need at least two coarse frames and one iteration")
        if self.max_fine_frames < 0 or self.max_videos < 0:
            raise ValueError("Frame/video limits must be nonnegative")
        if min(self.coarse_width, self.coarse_height, self.fine_max_pixels, self.max_new_tokens) <= 0:
            raise ValueError("Image dimensions, pixel and token budgets must be positive")
        for x in (self.sam_confidence, self.overlay_alpha, self.agreement_threshold):
            if not 0 <= x <= 1:
                raise ValueError("Thresholds must lie in [0, 1]")
