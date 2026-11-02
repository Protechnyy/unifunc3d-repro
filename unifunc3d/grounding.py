"""Prompts, explicit coordinate convention, temporal sampling and parsing."""
import json
import re

import numpy as np

TASK_TYPES = ("DIRECT_PART", "CONTROL_DEVICE", "CONNECTOR_INTERFACE")
# SceneFun3D task-2 affordance labels.
AFFORDANCE_TYPES = ("rotate", "key_press", "tip_push", "hook_pull", "pinch_pull",
                    "hook_turn", "foot_push", "plug_in", "unplug")


def sample_indices(length, count=64, iterations=4, iteration=0):
    """Equation (1), k=1..K, converted to zero-based and clipped at L-1."""
    if length < 1 or count < 2 or not 0 <= iteration < iterations:
        raise ValueError("Invalid sampling parameters")
    indices = np.floor(np.arange(count) * length / count +
                       (iteration + 1) * length / (iterations * count)).astype(int)
    return np.unique(np.clip(indices, 0, length - 1)).tolist()


def selection_prompt(task):
    return f"""You are an expert in task-driven 3D affordance grounding.

Given a natural language task and multiple RGB frames from the same indoor scene, identify the physical interaction required to complete the task and select informative frames.

Task: {task}

Complete the following steps:

1. Understand the task.
Classify the interaction into one of three types:
- DIRECT_PART: Manipulate a part of the target object itself, such as pulling a drawer handle or turning a door knob.
- CONTROL_DEVICE: Operate a control to change the target object's state, such as a light switch on the wall, a switch on a lamp, or a TV remote.
- CONNECTOR_INTERFACE: Connect or disconnect a device at a connection interface, such as a power socket.

2. Identify:
- Target object: the object whose state or connection the task changes, such as a drawer, a ceiling light, a TV or a lamp.
- Functional element: the specific element that must be physically manipulated to perform the action, such as a handle, knob, latch, switch, button, remote control or socket.
  It can be a part of the target object, such as a drawer handle, or a separate element, such as a wall switch for a ceiling light or a remote control for a TV.
  It is never the target object as a whole, and never the larger object it belongs to or is mounted on, such as a whole drawer, door, cabinet, window, wall or lamp.
  Write it as an element name only, such as "drawer handle", without position or color words.
- Affordance type: how the functional element is manipulated, one of {", ".join(AFFORDANCE_TYPES)}.
- Instance constraints: copy, word for word, every phrase from the task that identifies which instance is meant, such as order (top, bottom, third), side (left, right), color, and spatial relations to other objects. Return them as a list, or an empty list if the task has none.

Determine the target object, the functional element and the affordance type from the task before looking at the frames.
The functional element may differ from the target object and may be located away from it.

3. Select frames:
- Show the functional element of the correct instance.
- Avoid frames where the functional element is severely occluded or cut by the image boundary.
- If the instance constraints involve order, side or other objects, prefer frames that also show the similar objects or the other objects needed to check them.

Select at most 4 frames.
If no frame shows the functional element, return an empty frame_ids list. Do not replace it with another visible object.

Return valid JSON only:

{{
  "reasoning": "One or two sentences: which element must be manipulated and how, and which instance the constraints select.",
  "task_type": "{" | ".join(TASK_TYPES)}",
  "affordance_type": "{" | ".join(AFFORDANCE_TYPES)}",
  "target_object": "...",
  "functional_element": "...",
  "instance_constraints": [],
  "frame_ids": []
}}

Frame IDs are zero-based and must refer to the supplied images."""


def hypothesis_lines(hypothesis):
    return f"""Target object: {hypothesis["target_object"] or "None"}
Functional element: {hypothesis["functional_element"]}
Affordance type: {hypothesis["affordance_type"] or "None"}
Instance constraints: {"; ".join(hypothesis["instance_constraints"]) or "None"}"""


def localization_prompt(task, hypothesis):
    return f"""You are an expert in fine-grained visual grounding.

Task: {task}

{hypothesis_lines(hypothesis)}

You are given one RGB image from the scene.

Instructions:

1. Use the original task and the instance constraints to determine which instance of the functional element satisfies the task and to disambiguate similar objects.
2. If the instance constraints specify an order or side, first find all similar objects in the image, then choose the one that satisfies the constraints. If the image does not show enough of them to apply the constraints, output NONE.
3. Locate the functional element in the image. Its affordance type describes how it is manipulated and helps identify it.
4. Output one point strictly inside the visible functional element, not the target object or the larger object it belongs to or is mounted on.
5. If the correct functional element is not visible or cannot be distinguished reliably, output NONE.

Coordinates are normalized to [0, 1000].
The origin (0, 0) is the top-left corner of the image.
The bottom-right corner is (1000, 1000).

Output exactly one of:

<affordance> (x, y) </affordance>

<affordance> NONE </affordance>"""


def verify_prompt(task, hypothesis):
    return f"""You are verifying a fine-grained affordance segmentation.

Task: {task}

{hypothesis_lines(hypothesis)}

The RED highlighted region is a predicted segmentation mask.

Check whether the highlighted region accurately corresponds to the functional element required by the task.

Requirements:
- The mask must cover the functional element.
- Reject masks primarily covering the target object, the larger object the functional element belongs to or is mounted on, or surrounding surfaces.
- Reject masks covering unrelated objects or incorrect parts.
- Reject masks on an element that does not serve the task, even if its name matches the functional element.
- Reject masks on the same kind of element of a different instance than the instance constraints specify.

Output exactly one of:

<answer> YES </answer>

<answer> NO </answer>"""


def frame_ids(values, frame_count):
    if not isinstance(values, list):
        return [], [values]
    kept, dropped = [], []
    for value in values:
        if isinstance(value, str) and value.strip().isdigit():
            value = int(value)
        if type(value) is int and 0 <= value < frame_count:
            if value not in kept:
                kept.append(value)
        else:
            dropped.append(value)
    return kept, dropped


def parse_selection(text, frame_count, max_frames=4):
    result = {"error": None, "reasoning": "", "task_type": None, "task_type_valid": False,
              "affordance_type": None, "affordance_type_valid": False,
              "target_object": "", "functional_element": None, "instance_constraints": [],
              "frame_ids": [], "dropped_frame_ids": []}
    start = text.find("{")
    if start < 0:
        result["error"] = "no JSON object"
        return result
    try:
        value, _ = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError as exc:
        result["error"] = f"invalid JSON: {exc}"
        return result
    if not isinstance(value, dict):
        result["error"] = "JSON is not an object"
        return result
    element = value.get("functional_element")
    if not isinstance(element, str) or not element.strip():
        result["error"] = "missing functional_element"
        return result
    task_type = value.get("task_type")
    affordance_type = value.get("affordance_type")
    constraints = value.get("instance_constraints") or []
    if not isinstance(constraints, list):
        constraints = [constraints]
    selected, dropped = frame_ids(value.get("frame_ids", []), frame_count)
    # Extra frames beyond the prompt's limit are truncated in output order.
    result.update(reasoning=str(value.get("reasoning") or "").strip(),
                  task_type=task_type.strip().upper() if isinstance(task_type, str) else None,
                  affordance_type=affordance_type.strip().lower() if isinstance(affordance_type, str) else None,
                  target_object=str(value.get("target_object") or "").strip(),
                  functional_element=element.strip(),
                  instance_constraints=[str(x).strip() for x in constraints if str(x).strip()],
                  frame_ids=selected[:max_frames], dropped_frame_ids=dropped)
    # Both types are only recorded; an invalid value does not invalidate the selection.
    result["task_type_valid"] = result["task_type"] in TASK_TYPES
    result["affordance_type_valid"] = result["affordance_type"] in AFFORDANCE_TYPES
    # An empty list is a valid answer when no frame shows the functional element.
    if not result["frame_ids"] and dropped:
        result["error"] = "no valid frame_ids"
    return result


def parse_grounding(text):
    match = re.search(r"<affordance>(.*?)</affordance>", text, re.I | re.S)
    body = match.group(1) if match else text
    pts = re.findall(r"[\(\[]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\)\]]", body)
    return [(float(x), float(y)) for x, y in pts
            if 0 <= float(x) <= 1000 and 0 <= float(y) <= 1000]


def parse_verification(text):
    # Ambiguous/unparseable outputs fail closed, rather than silently accepting masks.
    answer = re.match(r"^\s*(?:<answer>\s*)?(YES|NO)\b", text, re.I)
    return bool(answer and answer.group(1).upper() == "YES")
