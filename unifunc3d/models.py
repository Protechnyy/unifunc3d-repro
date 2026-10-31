"""Local Qwen3-VL-8B and the official SAM3 interactive (point prompt) API."""
import logging
import time

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


class Qwen:
    def __init__(self, config):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.config = config
        torch.manual_seed(config.seed)
        self.processor = AutoProcessor.from_pretrained(config.qwen_model, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.qwen_model, dtype=torch.bfloat16, device_map="cuda",
            attn_implementation=config.attention, local_files_only=True).eval()
        if config.linear_patch_embed:
            from .torch_utils import LinearPatchEmbedding
            visual = self.model.model.visual
            visual.patch_embed = LinearPatchEmbedding(visual.patch_embed)
        from .feature_cache import VisualFeatureCache
        self.visual_cache = VisualFeatureCache(self.model.model.get_image_features)
        self.model.model.get_image_features = self.visual_cache

    def ask(self, images, prompt, tagged=False, max_pixels=None):
        import torch
        from .feature_cache import image_key
        # Coarse image groups repeat across descriptions within the same scene.
        # Fine/overlay images bypass this cache to preserve space for those groups.
        self.visual_cache.key = image_key(images, max_pixels) if tagged else None
        content = []
        for i in range(len(images)):
            if tagged:
                content.append({"type": "text", "text": f"<frame {i}>:"})
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt",
                                min_pixels=32 * 32 * 4,
                                max_pixels=max_pixels or self.config.fine_max_pixels).to("cuda")
        length = inputs.input_ids.shape[1]
        started = time.monotonic()
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens,
                                         do_sample=False, use_cache=True)
        answer = self.processor.batch_decode(output[:, length:], skip_special_tokens=True)[0].strip()
        log.info("Qwen %d images / %d tokens / %.1fs: %s", len(images), length,
                 time.monotonic() - started, answer)
        return {"text": answer, "input_tokens": length, "seconds": time.monotonic() - started}



class SAM3:
    def __init__(self, config):
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        self.config = config
        self.model = build_sam3_image_model(checkpoint_path=config.sam_checkpoint,
                                          load_from_HF=False, enable_inst_interactivity=True,
                                          device="cuda", compile=False)
        self.processor = Sam3Processor(self.model, confidence_threshold=config.sam_confidence)

    def segment(self, image, points):
        import torch
        result = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.processor.set_image(image)
            for x, y in points:
                pixel = [min(x / 1000 * image.width, image.width - 1),
                         min(y / 1000 * image.height, image.height - 1)]
                masks, scores, _ = self.model.predict_inst(
                    state, point_coords=np.array([pixel], dtype=np.float32),
                    point_labels=np.array([1], dtype=np.int32), multimask_output=True)
                # One mask per point, choosing SAM's highest predicted IoU.
                best = int(np.argmax(scores))
                score = float(scores[best])
                if score >= self.config.sam_confidence:
                    result.append({"point": [x, y], "score": score, "mask": masks[best].astype(bool)})
        return result


def overlay(image, mask, alpha=.7):
    arr = np.asarray(image).copy()
    arr[mask] = ((1 - alpha) * arr[mask] + alpha * np.array([255, 0, 0])).astype(np.uint8)
    return Image.fromarray(arr)
