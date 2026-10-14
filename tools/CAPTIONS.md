# Sinh caption bằng Qwen2.5-VL

Chạy trong môi trường đã cài các dependency của project (bao gồm `opencv-python`)
và các gói trong `requirements.txt`, với PyTorch hỗ trợ CUDA:

```bash
python -m pip install -r requirements.txt
python tools/generate_captions.py
```

Mặc định chạy tất cả dataset đã đăng ký trong `dataset.dataset_dict` có
`meta.json` tại root hiện tại. Lần đầu cần tải checkpoint
`Qwen/Qwen2.5-VL-7B-Instruct`; các lần sau dùng cache local. Không gọi API.

```bash
python tools/generate_captions.py --datasets sdd mpdd --dry-run
python tools/generate_captions.py --datasets mvtec --classes bottle cable
python tools/generate_captions.py --datasets mpdd --chunk-size 2
```

Script gọi `dataset.get_data(..., training=False)` và dùng `meta_info` từ loader.
Theo `BaseDataset`, cả training/testing hiện chỉ dùng split `test`.
Class và root lấy từ registry; nhãn lấy từ `anomaly` (`0` normal, `1` anomalous),
đường dẫn lấy từ `root / img_path`. Không đoán nhãn từ tên thư mục.
Mapping `sdd` hiện là `RSDD`, class `SDD`, theo `dataset/sdd.py`.

Mỗi **class và nhóm nhãn** lấy tối đa 30 ảnh khác nhau, chọn có seed và
phân bổ vòng tròn qua `specie_name` để bao phủ các loại lỗi trong metadata.
Nếu nhóm ít hơn 30 ảnh thì lấy toàn bộ; nếu không có ảnh thì báo lỗi trước
khi load model. Khi metadata không phân biệt loại lỗi, lấy mẫu ngẫu nhiên có seed.

Qwen được load một lần, BF16, SDPA, trên `cuda:0`. Mỗi lần chỉ đưa một chunk
lên GPU: mặc định 4 ảnh, tối đa 512 visual tokens/ảnh. Tensor và KV cache của
lượt gọi được giải phóng khi lượt gọi kết thúc; model ở lại GPU. Có thể giảm
`--chunk-size` nếu VRAM đang được dùng bởi chương trình khác.

Mỗi chunk được tổng hợp thành các đặc trưng quan sát được. Sau đó một lượt
text-only dùng ghi nhận từ **tất cả chunk** để tạo 20 caption tiếng Anh bổ sung
thông tin cho nhau. Ghi nhận normal được cung cấp khi quan sát anomalous để
giúp mô tả khác biệt thị giác. Prompt yêu cầu giữ các biến thể hiếm, không suy
diễn từ nhãn, không đoán chất liệu/diagnosis và không đổi từ đồng nghĩa để tạo
đa dạng giả. Đây là tổng hợp trên mẫu ảnh, không phải mô tả từng ảnh.
`NORMAL_PROMPTS` và `ANOMALOUS_PROMPTS` là hai bộ prompt riêng, mỗi bộ gồm
prompt quan sát ảnh và prompt tổng hợp caption. Normal mô tả đặc trưng thị giác
của ảnh normal; chỉ prompt anomalous nhận ghi nhận normal để đối chiếu.
Với anomalous, mỗi ghi nhận và caption phải tập trung vào bất thường quan sát
được; đặc trưng nền chỉ dùng để định vị hoặc giải thích defect. Ghi nhận chưa
nhìn rõ defect không được chuyển thành caption khẳng định ảnh không có lỗi.

Kết quả nằm trong `captions/<dataset>/<class>/normal.txt` và `anomalous.txt`,
chạy lại sẽ ghi đè file của class đó.

```bash
python -m unittest discover -s tests -p test_generate_captions.py
```

Cách nạp model, xử lý nhiều ảnh và giới hạn pixel theo
[model card chính thức của Qwen](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).
