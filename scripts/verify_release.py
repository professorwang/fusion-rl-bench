"""verify_release.py — v0.2.0 冻结资产一键验证（阶段 B5 外部复现包）

第三方复现入口（全流程见 docs/REPRODUCE.md）：

    python scripts/verify_release.py                    # 下载资产 → 哈希核对 → 评估 → 对照
    python scripts/verify_release.py --skip-download    # 资产已就位，只核对+评估
    python scripts/verify_release.py --skip-eval        # 只下载+核对哈希，不跑评估
    python scripts/verify_release.py --second-batch     # 追加种子 3010–3019 第二批对照

流程：
1. 从 GitHub Release v0.2.0 下载全部冻结资产（清单取自仓库根 SHA256SUMS.txt；
   已存在且哈希正确的文件自动跳过，幂等可重跑）；
2. 逐文件 SHA256 对照 SHA256SUMS.txt；
3. 用冻结 bc_policy.pt / ppo_final.zip 跑 scripts/fair_eval.py
   （种子 3000–3009、PD Kp=1 Kd=0.3、配对、动作后采样——与冻结口径一致）；
4. 与冻结 fair_eval_v013_s3000.json 逐控制器对照，输出 通过/漂移/失败 表。

发布说明：Release v0.2.0 当前仅含 SHA256SUMS.txt 所列 24 个文件中的 19 个；
B3 第二批消融产物（ablation_obs/horizon/transfer 的 json+png，共 6 个）未上传
发布。对这类文件脚本自动回退到仓库内 results/、assets/ 的同名文件做哈希核对
（状态记为"仓库内匹配(发布缺失)"），哈希覆盖率保持 24/24。

判据（跨平台数值漂移已证实，同一冻结权重+同种子，Windows↔macOS 相对偏差
最大约 ×1.8——开环不稳定通道放大数值细节，见 docs/SETUP.md 排坑 16）：
- 通过 PASS：资产哈希全部匹配；各控制器完整回合数与冻结值一致；
  mae_z_cm 排名一致（PD < BC < PPO）；且各误差指标相对比值在容差带
  [1/tol, tol] 内（默认 tol=2.0，覆盖已观测漂移上限 1.8）。
- 漂移 DRIFT：排名与完整率一致，但个别数值超出容差带——结论方向不变，
  属已证实的跨平台现象，退出码仍为 0，但会醒目标注供人工复核。
- 失败 FAIL：哈希不匹配 / 控制器缺失 / 完整率不一致 / 排名改变。退出码 1。

注意：验收判据是**排名、量级与结论方向一致**，不要求逐位一致。
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASE_TAG = "v0.2.0"
DOWNLOAD_BASE = ("https://github.com/professorwang/fusion-rl-bench/"
                 f"releases/download/{RELEASE_TAG}")
SUMS_FILE = os.path.join(REPO_ROOT, "SHA256SUMS.txt")

FROZEN_EVAL_S3000 = "fair_eval_v013_s3000.json"
FROZEN_EVAL_S3010 = "fair_eval_v013_s3010.json"
COMPARE_METRICS = ["mae_z_cm", "mae_joint_cm", "final_joint_cm"]


def load_sums(path):
    sums = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            h, name = line.split(None, 1)
            sums[name.strip()] = h.strip()
    return sums


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name, dest, timeout=300):
    """下载单个资产；404（发布中不存在）返回 False，其他错误照常抛出。"""
    url = f"{DOWNLOAD_BASE}/{name}"
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "fusion-rl-bench-verify"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    except urllib.error.HTTPError as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        if e.code == 404:
            return False
        raise
    os.replace(tmp, dest)
    return True


# 发布资产缺失时的仓库内回退核对目录（v0.2.0 发布有 19/24 资产；
# B3 第二批消融的 6 个文件仅在仓库 results/、assets/ 中，哈希一致）
LOCAL_CANDIDATES = ["results", "assets", "runs"]


def find_local(name):
    for d in LOCAL_CANDIDATES:
        p = os.path.join(REPO_ROOT, d, name)
        if os.path.exists(p):
            return p
    return None


def step_download_and_hash(sums, assets_dir, skip_download):
    print(f"\n== 1/3 冻结资产下载与 SHA256 核对（{len(sums)} 个文件，{RELEASE_TAG}）==")
    os.makedirs(assets_dir, exist_ok=True)
    rows, n_ok, n_release_missing = [], 0, 0
    for name, want in sums.items():
        dest = os.path.join(assets_dir, name)
        if not skip_download and not (
                os.path.exists(dest) and sha256_of(dest) == want):
            print(f"  下载 {name} …", flush=True)
            if not download(name, dest):
                n_release_missing += 1
                print(f"  [发布缺失] {name} 不在 Release {RELEASE_TAG} 资产中，"
                      "回退仓库内文件核对")
        src, got = None, None
        if os.path.exists(dest):
            src, got = "release", sha256_of(dest)
        else:
            local = find_local(name)
            if local is not None:
                src = "repo:" + os.path.relpath(local, REPO_ROOT)
                got = sha256_of(local)
        if got is None:
            rows.append((name, want, None, "缺失", None))
            continue
        ok = got == want
        n_ok += ok
        if ok:
            status = "匹配" if src == "release" else "仓库内匹配(发布缺失)"
        else:
            status = "不匹配"
        rows.append((name, want, got, status, src))
        if not ok:
            print(f"  [不匹配] {name}（{src}）\n    期望 {want}\n    实际 {got}")
    print(f"  SHA256 核对：{n_ok}/{len(sums)} 匹配"
          + (f"（其中 {n_release_missing} 个发布缺失、以仓库内文件核对）"
             if n_release_missing else ""))
    return rows, n_ok == len(sums)


def env_info():
    from importlib import metadata
    pkgs = ["numpy", "scipy", "torch", "stable_baselines3", "gymnasium",
            "freegsnke", "freegs4e", "numba", "llvmlite", "matplotlib"]
    vers = {}
    for p in pkgs:
        try:
            vers[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            vers[p] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": vers,
    }


def check_freegsnke_repo():
    repo = os.environ.get(
        "FREEGSNKE_REPO",
        os.path.join(REPO_ROOT, "envs", "..", "..", "vendor", "freegsnke"))
    repo = os.path.abspath(repo)
    need = [os.path.join(repo, "machine_configs", "MAST-U"),
            os.path.join(repo, "examples", "data",
                         "simple_diverted_currents_PaxisIp.pk")]
    missing = [p for p in need if not os.path.exists(p)]
    if missing:
        print("\n[失败] FreeGSNKE 装置数据未找到，以下路径不存在：")
        for p in missing:
            print(f"    {p}")
        print("  请按 docs/REPRODUCE.md 第 3 步克隆 FreeGSNKE 源码并设置 "
              "FREEGSNKE_REPO 环境变量。")
        return None
    print(f"  FreeGSNKE 数据目录：{repo}")
    return repo


def step_eval(assets_dir, seed0, episodes, tag):
    """跑一批 fair_eval，返回新鲜结果 JSON 路径。"""
    print(f"\n== 2/3 冻结模型配对评估（种子 {seed0}–{seed0 + episodes - 1}，"
          f"{episodes} 回合 × PD/BC/PPO）==")
    if check_freegsnke_repo() is None:
        return None
    bc = os.path.join(assets_dir, "bc_policy.pt")
    ppo = os.path.join(assets_dir, "ppo_final.zip")
    out = os.path.join(assets_dir, f"fair_eval_verify_{tag}.json")
    cmd = [sys.executable, os.path.join(REPO_ROOT, "scripts", "fair_eval.py"),
           "--episodes", str(episodes), "--seed0", str(seed0),
           "--pd-kp", "1", "--pd-kd", "0.3",
           "--bc", bc, "--ppo", ppo, "--out", out]
    print("  命令：" + " ".join(cmd), flush=True)
    print("  （首次 env.reset 需重建基线平衡 + numba JIT，约 1–3 分钟，"
          "之后每步约 16–50ms）", flush=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")  # Windows GBK 控制台排坑
    r = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    if r.returncode != 0:
        print(f"  [失败] fair_eval.py 退出码 {r.returncode}")
        return None
    return out


def verdict(frozen, fresh, tol):
    """对照一个种子批次的冻结/新鲜结果，返回 (每控制器行, 整体判定)。"""
    rows = []
    f_sum = {k: v["summary"] for k, v in frozen["results"].items()}
    n_sum = {k: v["summary"] for k, v in fresh["results"].items()}
    missing = [k for k in f_sum if k not in n_sum]
    if missing:
        return [{"controller": k, "status": "失败", "note": "新鲜结果缺失"}
                for k in missing], "失败"

    order_f = sorted(f_sum, key=lambda k: f_sum[k]["mae_z_cm"])
    order_n = sorted(n_sum, key=lambda k: n_sum[k]["mae_z_cm"])
    ranking_ok = order_f == order_n

    any_drift, any_fail = False, not ranking_ok
    for name in order_f:
        fs, ns = f_sum[name], n_sum[name]
        complete_ok = (fs["complete"] == ns["complete"]
                       and fs["valid_samples"] == ns["valid_samples"])
        if not complete_ok:
            any_fail = True
        ratios, worst = {}, None
        for m in COMPARE_METRICS:
            fv, nv = fs[m], ns[m]
            if fv is None or nv is None or fv == 0:
                ratios[m] = None
                continue
            r = nv / fv
            ratios[m] = r
            if not (1.0 / tol <= r <= tol):
                any_drift = True
                if worst is None or abs(r - 1) > abs(worst[1] - 1):
                    worst = (m, r)
        status = "通过"
        if not complete_ok:
            status = "失败"
        elif worst is not None:
            status = "漂移"
        rows.append({
            "controller": name,
            "complete_fresh": f"{ns['complete']}/{ns['valid_samples']}/{ns['episodes']}",
            "complete_frozen": f"{fs['complete']}/{fs['valid_samples']}/{fs['episodes']}",
            "mae_z_cm_frozen": fs["mae_z_cm"],
            "mae_z_cm_fresh": ns["mae_z_cm"],
            "ratios_fresh_over_frozen": ratios,
            "status": status,
        })
    if any_fail:
        overall = "失败"
    elif any_drift:
        overall = "漂移"
    else:
        overall = "通过"
    return rows, overall, ranking_ok, order_f, order_n


def print_table(tag, rows, overall, tol):
    print(f"\n== 3/3 数值对照（{tag}，容差带 [1/{tol:g}, {tol:g}]，比值=新鲜/冻结）==")
    print(f"  {'控制器':<34}{'完整(新/冻)':>16}{'MAE_Z 冻结':>12}"
          f"{'MAE_Z 新鲜':>12}{'比值':>8}  判定")
    for r in rows:
        ratio = (r["ratios_fresh_over_frozen"] or {}).get("mae_z_cm")
        ratio_s = f"{ratio:8.3f}" if ratio else "     n/a"
        print(f"  {r['controller']:<34}"
              f"{r['complete_fresh'] + ' / ' + r['complete_frozen']:>16}"
              f"{r['mae_z_cm_frozen']:>12.4f}{r['mae_z_cm_fresh']:>12.4f}"
              f"{ratio_s}  {r['status']}")
    print(f"  批次判定：{overall}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets-dir", default=os.path.join(
        REPO_ROOT, "release_assets", RELEASE_TAG),
        help="冻结资产目录（默认 release_assets/v0.2.0/，未纳入 git 跟踪）")
    ap.add_argument("--skip-download", action="store_true",
                    help="不下载，使用 --assets-dir 中已有文件")
    ap.add_argument("--skip-eval", action="store_true",
                    help="只做下载+哈希核对，不跑评估")
    ap.add_argument("--second-batch", action="store_true",
                    help="追加种子 3010–3019 批次，对照 fair_eval_v013_s3010.json")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--tol", type=float, default=2.0,
                    help="相对比值容差（默认 2.0，覆盖已观测跨平台漂移上限 ×1.8）")
    ap.add_argument("--out", default="",
                    help="验证报告 JSON 输出路径（默认 <assets-dir>/verify_report.json）")
    args = ap.parse_args()

    if not os.path.exists(SUMS_FILE):
        print(f"[失败] 未找到 {SUMS_FILE}——请在仓库根目录的克隆中运行本脚本。")
        return 1
    sums = load_sums(SUMS_FILE)
    info = env_info()
    print("环境：Python {python} | {platform} | {machine}".format(**info))
    print("依赖版本：" + ", ".join(
        f"{k}={v or '未装'}" for k, v in info["packages"].items()))

    hash_rows, hash_ok = step_download_and_hash(sums, args.assets_dir,
                                                args.skip_download)

    batches, overall = [], "通过" if hash_ok else "失败"
    if not args.skip_eval:
        plan = [(3000, FROZEN_EVAL_S3000, "s3000")]
        if args.second_batch:
            plan.append((3010, FROZEN_EVAL_S3010, "s3010"))
        for seed0, frozen_name, tag in plan:
            fresh_path = step_eval(args.assets_dir, seed0, args.episodes, tag)
            frozen_path = os.path.join(args.assets_dir, frozen_name)
            if fresh_path is None or not os.path.exists(frozen_path):
                batches.append({"tag": tag, "overall": "失败",
                                "note": "评估未运行或冻结对照缺失"})
                overall = "失败"
                continue
            with open(frozen_path, encoding="utf-8") as f:
                frozen = json.load(f)
            with open(fresh_path, encoding="utf-8") as f:
                fresh = json.load(f)
            rows, b_overall, ranking_ok, of, on = verdict(frozen, fresh, args.tol)
            print_table(tag, rows, b_overall, args.tol)
            if not ranking_ok:
                print(f"  [失败] 排名改变：冻结 {' < '.join(of)} vs 新鲜 {' < '.join(on)}")
            batches.append({"tag": tag, "seed0": seed0,
                            "episodes": args.episodes, "tol": args.tol,
                            "ranking_ok": ranking_ok,
                            "ranking_frozen": of, "ranking_fresh": on,
                            "controllers": rows, "overall": b_overall,
                            "fresh_file": fresh_path})
            if b_overall == "失败":
                overall = "失败"
            elif b_overall == "漂移" and overall != "失败":
                overall = "漂移"

    print(f"\n总体判定：{overall}"
          + ("（排名/完整率一致，数值漂移超容差带——跨平台已知现象，结论方向不变）"
             if overall == "漂移" else ""))
    print("判据说明：验收标准为排名、量级与结论方向一致，不要求逐位一致；"
          "已观测跨平台漂移上限约 ×1.8（docs/SETUP.md 排坑 16）。")

    report = {"release": RELEASE_TAG, "environment": info,
              "hash_check": {"total": len(hash_rows),
                             "matched": sum(1 for r in hash_rows
                                            if r[3] in ("匹配", "仓库内匹配(发布缺失)")),
                             "release_missing": sum(1 for r in hash_rows
                                                    if r[3] == "仓库内匹配(发布缺失)"),
                             "rows": [{"file": r[0], "expected": r[1],
                                       "actual": r[2], "status": r[3],
                                       "source": r[4]}
                                      for r in hash_rows]},
              "batches": batches, "overall": overall}
    out = args.out or os.path.join(args.assets_dir, "verify_report.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"验证报告已存 {out}")
    return 0 if overall in ("通过", "漂移") else 1


if __name__ == "__main__":
    sys.exit(main())
