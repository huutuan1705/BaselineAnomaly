import contextlib
import io
import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location(
    "caption_script", Path(__file__).resolve().parents[1] / "tools" / "generate_captions.py")
caption_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(caption_script)
MODEL, Qwen = caption_script.MODEL, caption_script.Qwen
describe, main = caption_script.describe, caption_script.main
parse_captions, select_images = caption_script.parse_captions, caption_script.select_images


def valid_output():
    return json.dumps([f"Visible feature {i}." for i in range(20)])


class CaptionTests(unittest.TestCase):
    def test_sampling_uses_labels_and_balances_metadata_species(self):
        rows = [{"img_path": f"arbitrary/{kind}/{i}.png", "anomaly": label,
                 "specie_name": kind}
                for label, kind, count in [(0, "not_good", 50), (1, "rare", 2),
                                           (1, "good", 80)] for i in range(count)]
        selected = select_images(rows + rows, 1)
        self.assertEqual(len(selected), 30)
        self.assertEqual(len({r["img_path"] for r in selected}), 30)
        self.assertEqual(sum(r["specie_name"] == "rare" for r in selected), 2)
        self.assertTrue(all(r["anomaly"] == 1 for r in selected))
        self.assertEqual(selected, select_images(list(reversed(rows)), 1))
        self.assertEqual(len(select_images(rows[:8], 0)), 8)
        with self.assertRaises(ValueError):
            select_images(rows[:8], 1)

    def test_chunk_notes_are_all_used_in_synthesis(self):
        qwen = Mock()
        qwen.ask.side_effect = ["fact A", "fact B", "fact C", valid_output()]
        paths = [Path(str(i)) for i in range(9)]
        with contextlib.redirect_stdout(io.StringIO()):
            captions, evidence = describe(qwen, paths, "test", "anomalous", 4, "smooth outline")
        self.assertEqual(len(captions), 20)
        calls = qwen.ask.call_args_list
        self.assertEqual([call.args[1] for call in calls[:3]], [paths[:4], paths[4:8], paths[8:]])
        self.assertTrue(all("smooth outline" in call.args[0] for call in calls[:3]))
        self.assertTrue(all(fact in evidence and fact in calls[3].args[0]
                            for fact in ["fact A", "fact B", "fact C"]))
        self.assertEqual(len(calls[3].args), 1)  # Text-only synthesis, no old image tensors.

    def test_qwen_load_and_chunk_forward(self):
        from PIL import Image

        torch = Mock()
        torch.cuda.is_available.return_value = True
        torch.inference_mode.side_effect = contextlib.nullcontext
        transformers = Mock()
        with patch.dict("sys.modules", {"torch": torch, "transformers": transformers}):
            qwen = Qwen(512 * 28 * 28)
        loader = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained
        loader.assert_called_once()
        self.assertEqual(loader.call_args.args[0], MODEL)
        self.assertEqual(loader.call_args.kwargs["device_map"], {"": "cuda:0"})
        inputs = Mock()
        inputs.input_ids.shape = (1, 8)
        inputs.keys.return_value = ["input_ids"]
        inputs.__getitem__ = Mock(return_value="tensor")
        qwen.processor.return_value.to.return_value = inputs
        output = Mock()
        output.__getitem__ = Mock(return_value="generated tokens")
        qwen.model.generate.return_value = output
        qwen.processor.decode.return_value = "captions"
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"{i}.png" for i in range(2)]
            for path in paths:
                Image.new("RGB", (32, 32)).save(path)
            self.assertEqual(qwen.ask("describe", paths), "captions")
            self.assertEqual(len(qwen.processor.call_args.kwargs["images"]), 2)
            qwen.processor.return_value.to.assert_called_with("cuda:0")
            qwen.ask("synthesize")
            self.assertIsNone(qwen.processor.call_args.kwargs["images"])
        output.__getitem__.assert_called_with((0, slice(8, None)))
        loader.assert_called_once()  # More requests do not reload the model.

    def test_main_uses_loader_paths_and_writes_both_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"img_path": "unusual-normal.png", "anomaly": 0},
                    {"img_path": "unusual-anomalous.png", "anomaly": 1}]
            for row in rows:
                (root / row["img_path"]).touch()
            loader = Mock(return_value=(["Class"], types.SimpleNamespace(meta_info={"Class": rows}), root))
            dataset = types.SimpleNamespace(dataset_dict={"example": (["Class"], None, root)}, get_data=loader)
            argv = ["generate_captions.py", "--datasets", "example", "--output", str(root / "captions")]
            with patch.dict("sys.modules", {"dataset": dataset}), patch("sys.argv", argv), \
                    patch.object(caption_script, "Qwen") as qwen, \
                    contextlib.redirect_stdout(io.StringIO()):
                qwen.return_value.ask.side_effect = ["normal evidence", valid_output(),
                                                      "anomalous evidence", valid_output()]
                main()
                qwen.assert_called_once()
                loader.assert_called_once_with("example", None, None, training=False)
                self.assertEqual(qwen.return_value.ask.call_args_list[0].args[1], [root / rows[0]["img_path"]])
                for state in ["normal", "anomalous"]:
                    lines = (root / "captions" / "example" / "Class" / f"{state}.txt").read_text().splitlines()
                    self.assertEqual(parse_captions(json.dumps(lines)), json.loads(valid_output()))
            with patch.dict("sys.modules", {"dataset": dataset}), patch("sys.argv", argv + ["--dry-run"]), \
                    patch.object(caption_script, "Qwen") as qwen, contextlib.redirect_stdout(io.StringIO()):
                main()
                qwen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
