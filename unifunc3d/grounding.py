"""Paper prompts, explicit coordinate convention, temporal sampling and parsing."""
import re

import numpy as np


def coarse_indices(length, count=64, iterations=4, iteration=0):
    """Equation (1), k=1..K, converted to zero-based and clipped at L-1."""
    if length < 1 or count < 2 or not 0 <= iteration < iterations:
        raise ValueError("Invalid sampling parameters")
    indices = np.floor(np.arange(count) * length / count +
                       (iteration + 1) * length / (iterations * count)).astype(int)
    return np.unique(np.clip(indices, 0, length - 1)).tolist()


def window_indices(timestamps, center, coarse_frames=64):
    timestamps = np.asarray(timestamps)
    radius = (timestamps[-1] - timestamps[0]) / (2 * (coarse_frames - 1))
    return np.flatnonzero(np.abs(timestamps - timestamps[center]) <= radius + 1e-7).tolist()


def coarse_prompt(task, part_aware=False):
    prompt = (f"Given these frames and the task: [{task}], complete three tasks:\n"
            "1. Identify the functional object needed to accomplish the task.\n"
            "2. Select the key frame that best shows the functional object.\n"
            "3. Identify a single affordance point (x, y) on the functional object.\n"
            "Output format: <affordance> functional object: ...; <frame n>: ...; (x, y) </affordance>\n"
            "Use the supplied zero-based frame tags. Coordinates are normalized to [0, 1000].")
    if part_aware:
        prompt += ("\nThe functional object is the concrete interactive part that a hand must touch "
                   "to perform the action, not its parent or containing object. For example, "
                   "opening a cabinet requires its handle or knob, and turning on a light "
                   "requires its switch. Preserve a specific part explicitly named in the task. "
                   "Name that part and select a frame where that part is most clearly visible.")
    return prompt


def fine_prompt(task, functional_object):
    return (f"Identify the affordance point on the [{functional_object}] in order to [{task}].\n"
            "Output format: <affordance> (x, y) </affordance>.\n"
            "Coordinates are normalized to [0, 1000]. If the target is not visible, "
            "output <affordance> NONE </affordance>.")


def verify_prompt(functional_object):
    return (f"Does the RED highlighted region show ONLY [{functional_object}]?\n"
            "Answer YES or NO. Reject regions that include the parent or containing object.")


def parse_grounding(text, frame_count=None):
    match = re.search(r"<affordance>(.*?)</affordance>", text, re.I | re.S)
    body = match.group(1) if match else text
    pts = re.findall(r"[\(\[]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\)\]]", body)
    points = [(float(x), float(y)) for x, y in pts
              if 0 <= float(x) <= 1000 and 0 <= float(y) <= 1000]
    obj = re.search(r"functional object\s*:\s*(.*?)(?:;|<frame|\n|$)", body, re.I)
    frame = re.search(r"<frame\s+(\d+)\s*>", body, re.I)
    result = {"points": points, "functional_object": obj.group(1).strip() if obj else None,
              "frame": int(frame.group(1)) if frame else None}
    if frame_count is not None and (result["frame"] is None or
                                   not 0 <= result["frame"] < frame_count):
        result["frame"] = None
    return result


def parse_verification(text):
    # Ambiguous/unparseable outputs fail closed, rather than silently accepting masks.
    answer = re.match(r"^\s*(?:<answer>\s*)?(YES|NO)\b", text, re.I)
    return bool(answer and answer.group(1).upper() == "YES")
