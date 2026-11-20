import io, os
from PIL import Image

src = r"C:\Users\chris\.dsh\attachments\v1\objects\36\3621e67adcbacf1c3280bf17e926576e448e832a29662d7fc9761423a16b1d77"
out_dir = r"I:\Fermi\quantum-atlas\assets"
os.makedirs(out_dir, exist_ok=True)

im = Image.open(src).convert("RGBA")
W, H = im.size
print("source", W, H)

# 方形裁切：以原子为中心，保留底部的地面光晕
side = min(W, H)
cx, cy = W // 2, int(H * 0.50)
left = max(0, min(W - side, cx - side // 2))
top = max(0, min(H - side, cy - side // 2))
sq = im.crop((left, top, left + side, top + side))
print("crop", (left, top, side))

# 主尺寸
for size in (512, 256, 128, 64, 32, 16):
    sq.resize((size, size), Image.LANCZOS).save(os.path.join(out_dir, f"icon-{size}.png"))

# Windows 多尺寸 ico（任务栏/资源管理器都取得到清晰尺寸）
icons = [sq.resize((s, s), Image.LANCZOS) for s in (256, 128, 64, 48, 32, 16)]
icons[0].save(os.path.join(out_dir, "icon.ico"), format="ICO", sizes=[(s, s) for s in (256, 128, 64, 48, 32, 16)])
print("written", sorted(os.listdir(out_dir)))
