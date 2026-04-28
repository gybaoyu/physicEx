import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def parse_pairs(line: str) -> np.ndarray:
    pairs = re.findall(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", line)
    if not pairs:
        return np.empty((0, 2), dtype=float)
    return np.array([(float(x), float(y)) for x, y in pairs], dtype=float)


def catmull_rom(points: np.ndarray, samples_per_seg: int = 25, closed: bool = False) -> np.ndarray:
    pts = np.array(points, dtype=float)
    if len(pts) < 2:
        return pts

    n = len(pts)
    tangents = np.zeros_like(pts)
    if closed:
        for i in range(n):
            tangents[i] = 0.5 * (pts[(i + 1) % n] - pts[(i - 1) % n])
        seg_count = n
    else:
        tangents[0] = pts[1] - pts[0]
        tangents[-1] = pts[-1] - pts[-2]
        for i in range(1, n - 1):
            tangents[i] = 0.5 * (pts[i + 1] - pts[i - 1])
        seg_count = n - 1

    out = []
    t_vals = np.linspace(0.0, 1.0, samples_per_seg, endpoint=False)
    for i in range(seg_count):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        m0 = tangents[i]
        m1 = tangents[(i + 1) % n]
        for t in t_vals:
            t2 = t * t
            t3 = t2 * t
            h00 = 2 * t3 - 3 * t2 + 1
            h10 = t3 - 2 * t2 + t
            h01 = -2 * t3 + 3 * t2
            h11 = t3 - t2
            out.append(h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1)

    if closed:
        out.append(pts[0])
    else:
        out.append(pts[-1])
    return np.array(out, dtype=float)


def load_data(data_file: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = data_file.read_text(encoding="utf-8").splitlines()
    parsed_groups = [parse_pairs(ln) for ln in lines if "," in ln]
    parsed_groups = [arr for arr in parsed_groups if len(arr) > 0]
    if len(parsed_groups) < 2:
        raise ValueError("未在数据文件中找到两组坐标数据（18点回线 + 端点数据）。")

    major_loop = parsed_groups[0]
    endpoints = parsed_groups[1]
    return major_loop, endpoints


def order_clockwise_from_upper_right(points: np.ndarray) -> np.ndarray:
    pts = np.array(points, dtype=float)
    if len(pts) < 3:
        return pts

    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(-angles)
    ordered = pts[order]

    start_idx = np.argmax(ordered[:, 0] + ordered[:, 1])
    ordered = np.roll(ordered, -start_idx, axis=0)
    return ordered


def unique_by_round(points: np.ndarray, ndigits: int = 6) -> np.ndarray:
    pts = np.array(points, dtype=float)
    if len(pts) == 0:
        return pts
    seen = set()
    unique = []
    for p in pts:
        key = (round(float(p[0]), ndigits), round(float(p[1]), ndigits))
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return np.array(unique, dtype=float)


def nice_step(raw_step: float) -> float:
    if raw_step <= 0:
        return 1.0
    exp = np.floor(np.log10(raw_step))
    frac = raw_step / (10**exp)
    if frac <= 1:
        base = 1
    elif frac <= 2:
        base = 2
    elif frac <= 2.5:
        base = 2.5
    elif frac <= 5:
        base = 5
    else:
        base = 10
    return float(base * (10**exp))


def build_magnetization_branches(major_loop: np.ndarray, endpoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_idx = np.argmax(major_loop[:, 0])
    min_idx = np.argmin(major_loop[:, 0])
    major_pos = major_loop[max_idx]
    major_neg = major_loop[min_idx]

    right_branch = endpoints[endpoints[:, 0] >= 0]
    left_branch = endpoints[endpoints[:, 0] <= 0]

    right_branch = unique_by_round(np.vstack([major_pos, right_branch]))
    left_branch = unique_by_round(np.vstack([major_neg, left_branch]))

    right_amp = np.hypot(right_branch[:, 0], right_branch[:, 1])
    left_amp = np.hypot(left_branch[:, 0], left_branch[:, 1])
    right_branch = right_branch[np.argsort(-right_amp)]
    left_branch = left_branch[np.argsort(-left_amp)]

    return right_branch, left_branch


def plot_curves(major_loop: np.ndarray, endpoints: np.ndarray, out_file: Path | None = None, show: bool = True) -> None:
    loop_ordered = order_clockwise_from_upper_right(major_loop)
    loop_smooth = catmull_rom(loop_ordered, samples_per_seg=30, closed=True)

    right_branch, left_branch = build_magnetization_branches(loop_ordered, endpoints)
    right_smooth = catmull_rom(right_branch, samples_per_seg=30, closed=False)
    left_smooth = catmull_rom(left_branch, samples_per_seg=30, closed=False)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(loop_smooth[:, 0], loop_smooth[:, 1], color="#1565C0", linewidth=2.2, label="磁滞回线（平滑且过点）")
    ax.scatter(loop_ordered[:, 0], loop_ordered[:, 1], color="#1565C0", s=20, alpha=0.55)

    ax.plot(right_smooth[:, 0], right_smooth[:, 1], color="#D32F2F", linewidth=2.2, label="磁化曲线（右支，过点）")
    ax.plot(left_smooth[:, 0], left_smooth[:, 1], color="#D32F2F", linewidth=2.2, linestyle="--", label="磁化曲线（左支，过点）")
    ax.scatter(endpoints[:, 0], endpoints[:, 1], color="#D32F2F", s=18, alpha=0.6)

    ax.axhline(0, color="gray", linewidth=1.0, alpha=0.7)
    ax.axvline(0, color="gray", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("X（相对 H）")
    ax.set_ylabel("Y（相对 B）")
    ax.set_title("磁滞回线与磁化曲线（同图）")

    all_pts = np.vstack([loop_ordered, endpoints])
    x_range = np.max(all_pts[:, 0]) - np.min(all_pts[:, 0])
    y_range = np.max(all_pts[:, 1]) - np.min(all_pts[:, 1])
    x_step = nice_step(x_range / 7)
    y_step = nice_step(max(y_range / 7, 0.7 * x_step))
    ax.xaxis.set_major_locator(MultipleLocator(x_step))
    ax.yaxis.set_major_locator(MultipleLocator(y_step))

    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    if out_file is not None:
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        print(f"已保存图像：{out_file}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="根据实验数据绘制磁滞回线与磁化曲线（同图）。")
    parser.add_argument(
        "--data",
        default="实验数据.txt",
        help="数据文件路径（默认：实验数据.txt）",
    )
    parser.add_argument(
        "--out",
        default="磁滞回线_磁化曲线.png",
        help="输出图片路径（默认：磁滞回线_磁化曲线.png）",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="仅保存图片，不弹出窗口。",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"未找到数据文件：{data_path}")

    major_loop, endpoints = load_data(data_path)
    plot_curves(major_loop, endpoints, Path(args.out), show=not args.no_show)


if __name__ == "__main__":
    main()
