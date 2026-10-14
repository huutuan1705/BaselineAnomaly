"""Generate class-level visual descriptions using local Qwen2.5-VL."""
import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
VISUAL_RULES = """Use only clearly visible evidence; omit uncertain details.
Describe the collection, not individual images. Do not infer attributes from
class names or labels, causes, function, diagnoses, or invisible materials.
Distinguish common traits from occasional variants; do not claim every sample
has every feature. Never invent a feature to increase diversity."""

NORMAL_PROMPTS = (
    """Inspect the NORMAL collection of class {class_name!r}.
Describe visible object shape, structure, texture, surface finish, color,
pattern, alignment, and intact boundaries. Preserve shared traits and rare
visual variations. Focus on concrete appearance rather than generic claims
that the images have no defects. Do not speculate about possible defects.
{rules}
List up to 16 concise visual facts, grouped by attribute, at most 200 words.""",
    """Write exactly 20 English captions describing the NORMAL collection
of class {class_name!r}, using ONLY the normal observations below.
Cover complementary shape, structure, texture, surface, color, pattern,
and regularity. Each caption must add a distinct fact or meaningful combination,
not merely synonyms. Cover all chunks and preserve occasional variations.
Describe concrete visible traits rather than repeating 'no defects'.
{rules}
The word normal is optional. Do not refer to image/chunk numbers.
Each caption must contain at most 20 words. Return ONLY a JSON array of strings.
If evidence is insufficient, return fewer instead of inventing.
NORMAL OBSERVATIONS:\n{evidence}""",
)

ANOMALOUS_PROMPTS = (
    """Inspect the ANOMALOUS collection of class {class_name!r}.
Extract ONLY visible defects, irregularities, and departures from the surrounding
pattern or normal reference. Every visual fact must identify an observed
abnormal feature, describing its kind, appearance, location, extent, contrast,
texture disruption, or structural change where visible. Preserve rare defects.
Ordinary object or background traits may only locate or explain a defect;
do not list generic color, uniform texture, intact surfaces, or clean edges.
Labels are not proof of a visible defect. If none is clearly observed, record
uncertainty without claiming that the images are defect-free or inventing a defect.
{rules}
List up to 16 concise defect facts, at most 200 words.
NORMAL REFERENCE (comparison only, not defect evidence):\n{baseline}""",
    """Write exactly 20 English captions describing visible ABNORMALITIES
in class {class_name!r}, using ONLY the anomalous observations below.
EVERY caption must identify a concrete observed defect or deviation.
Diversify observed defect kinds, shapes, locations, extent, color contrast,
texture disruption, and structural changes. Background traits only serve
to locate or explain the abnormality. Exclude generic object descriptions,
intact appearance, and statements that no defect is visible. Uncertainty notes
are not defect evidence and must not become captions. Do not copy normal traits.
Each caption must add a distinct fact or meaningful combination, not synonyms.
Cover defect evidence from all chunks, including rare variations.
{rules}
The words abnormal, anomalous, and defective are optional.
Do not refer to image/chunk numbers. Each caption must contain at most 20 words.
Return ONLY a JSON array of strings. If defect evidence is insufficient,
return fewer captions instead of filling with normal traits or invented defects.
ANOMALOUS OBSERVATIONS:\n{evidence}""",
)


def select_images(rows, label, seed=42):
    """Take up to 30 unique examples, round-robin across metadata species."""
    groups, seen = defaultdict(list), set()
    for row in sorted(rows, key=lambda row: row["img_path"]):
        if row["anomaly"] == label and row["img_path"] not in seen:
            groups[row.get("specie_name", "")].append(row)
            seen.add(row["img_path"])
    rng = random.Random(seed)
    pools = [groups[key] for key in sorted(groups)]
    for pool in pools:
        rng.shuffle(pool)
    selected = []
    while pools and len(selected) < 30:
        for pool in pools:
            selected.append(pool.pop())
            if len(selected) == 30:
                break
        pools = [pool for pool in pools if pool]
    if not selected:
        raise ValueError(f"No images with anomaly={label}; cannot invent captions.")
    return selected


def parse_captions(raw):
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    captions = json.loads(raw)
    captions = [" ".join(c.split()) for c in captions]
    return captions


class Qwen:
    def __init__(self, max_pixels):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Run in a CUDA-enabled environment.")
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            MODEL, min_pixels=128 * 28 * 28, max_pixels=max_pixels)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
            attn_implementation="sdpa").eval()

    def ask(self, prompt, paths=()):
        from PIL import Image

        images = []
        try:
            for path in paths:
                # Match BaseDataset's TIFF decoder; no transforms/augmentation.
                if str(path).endswith(".tif"):
                    import cv2
                    pixels = cv2.imread(str(path))
                    if pixels is None:
                        raise ValueError(f"Cannot decode image: {path}")
                    images.append(Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)))
                else:
                    with Image.open(path) as image:
                        images.append(image.convert("RGB"))
            content = [{"type": "image", "image": image} for image in images]
            content.append({"type": "text", "text": prompt})
            text = self.processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False,
                add_generation_prompt=True)
            inputs = self.processor(text=[text], images=images or None,
                                    return_tensors="pt").to("cuda:0")
            with self.torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            return self.processor.decode(output[0, inputs.input_ids.shape[1]:],
                                         skip_special_tokens=True).strip()
        finally:
            # Release this chunk's tensors/KV cache; keep the model resident.
            for image in images:
                image.close()


def describe(qwen, paths, class_name, state, chunk_size, baseline=""):
    observation_prompt, caption_prompt = NORMAL_PROMPTS if state == "normal" else ANOMALOUS_PROMPTS
    prompt = observation_prompt.format(class_name=class_name, rules=VISUAL_RULES,
                                       baseline=baseline or "(not provided)")
    notes = []
    for start in range(0, len(paths), chunk_size):
        chunk = paths[start:start + chunk_size]
        print(f"  {state}: images {start + 1}-{start + len(chunk)}/{len(paths)}", flush=True)
        notes.append(qwen.ask(prompt, chunk))
    evidence = "\n\n".join(f"Chunk {i + 1}: {note}" for i, note in enumerate(notes))
    prompt = caption_prompt.format(class_name=class_name, rules=VISUAL_RULES, evidence=evidence)
    raw = qwen.ask(prompt)
    return parse_captions(raw), evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", help="Dataset keys from dataset.dataset_dict")
    parser.add_argument("--classes", nargs="+", help="Only these exact loader class names")
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "captions")
    parser.add_argument("--dry-run", action="store_true", help="Check metadata and image paths without Qwen")
    args = parser.parse_args()
    if args.chunk_size < 1 or args.max_pixels < 128 * 28 * 28:
        parser.error("chunk-size must be positive and max-pixels >= 100352")
    from dataset import dataset_dict, get_data

    names = args.datasets if args.datasets is not None else [
        name for name, (_, _, root) in dataset_dict.items() if (Path(root) / "meta.json").is_file()]
    if not names or any(name not in dataset_dict for name in names):
        parser.error(f"Choose available dataset keys: {', '.join(dataset_dict)}")
    known_classes = {cls for name in names for cls in dataset_dict[name][0]}
    if args.classes and set(args.classes) - known_classes:
        parser.error(f"Unknown classes: {sorted(set(args.classes) - known_classes)}")
    jobs = []
    for name in dict.fromkeys(names):
        classes, dataset, root = get_data(name, None, None, training=False)
        for cls in classes:
            if args.classes and cls not in args.classes:
                continue
            groups = [select_images(dataset.meta_info[cls], label, args.seed) for label in (0, 1)]
            paths = [[Path(root) / row["img_path"] for row in group] for group in groups]
            for path in paths[0] + paths[1]:
                if not path.is_file():
                    raise FileNotFoundError(path)
            print(f"{name}/{cls}: {len(paths[0])} normal, {len(paths[1])} anomalous (test)")
            jobs.append((name, cls, paths))
    if args.dry_run:
        return
    qwen = Qwen(args.max_pixels)  # Exactly one model load for every dataset/class.
    for name, cls, paths in jobs:
        print(f"Generating {name}/{cls}", flush=True)
        normal, baseline = describe(qwen, paths[0], cls, "normal", args.chunk_size)
        anomalous, _ = describe(qwen, paths[1], cls, "anomalous", args.chunk_size, baseline)
        directory = args.output / name / cls
        directory.mkdir(parents=True, exist_ok=True)
        for state, captions in (("normal", normal), ("anomalous", anomalous)):
            (directory / f"{state}.txt").write_text("\n".join(captions) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
