"""
本地 Sionna RT site-specific 实验脚本
环境：sionna-new（Sionna 1.2.2 + TF 2.21，Python 3.10）

运行方式：
    conda run -n sionna-new python run_rt_local.py --scene munich
    conda run -n sionna-new python run_rt_local.py --scene etoile
    conda run -n sionna-new python run_rt_local.py --scene sydney --sydney-zip /path/to/sydney_scene.zip

输出：
    checkpoints/image-jscc/eval/site_specific_results_{scene}.json
    checkpoints/image-jscc/eval/site_specific_results_{scene}.png
    checkpoints/image-jscc/eval/site_specific_heatmap_{scene}.png
"""

import argparse, glob, importlib, json, os, re, sys, types, zipfile
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.interpolate import griddata

os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
os.environ.setdefault('DRJIT_LIBLLVM_PATH', '/usr/lib/llvm-14/lib/libLLVM-14.so')

# sionna.rt 必须在 TF 之前 import，否则 Dr.Jit 初始化与 TF GPU 冲突导致 segfault
import sionna.rt as srt
from sionna.rt import (load_scene, PlanarArray, Transmitter, Receiver,
                       Camera, PathSolver, RadioMapSolver)

import tensorflow as tf
for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)

WORK_DIR   = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR   = os.path.join(WORK_DIR, 'checkpoints/image-jscc/eval')
CACHE_DIR  = os.path.join(WORK_DIR, 'checkpoints/image-jscc/rt_cache')
os.makedirs(EVAL_DIR,  exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

sys.path.insert(0, WORK_DIR)

# 拦截 models/__init__.py 的 BART import 链（sionna-new 没有 transformers）
_models_stub = types.ModuleType('models')
_models_stub.__path__    = [os.path.join(WORK_DIR, 'models')]
_models_stub.__package__ = 'models'
sys.modules['models'] = _models_stub

# 注入 models.channels stub（原版依赖 sionna.channel，在 Sionna 1.x 已移除）
import tensorflow as _tf

class _ChannelAWGN(_tf.keras.layers.Layer):
    def call(self, x, ebno_db=10.0, training=False, ebno_db_override=None):
        if ebno_db_override is not None:
            ebno_db = ebno_db_override
        ebno_linear = _tf.cast(10.0 ** (float(ebno_db) / 10.0), _tf.float32)
        noise_std   = _tf.sqrt(_tf.reduce_mean(_tf.square(x)) / (2.0 * ebno_linear) + 1e-10)
        return x + _tf.random.normal(_tf.shape(x), stddev=noise_std)

class _ChannelCDL(_tf.keras.layers.Layer):
    def __init__(self, *args, ebno_db_min=5, ebno_db_max=15, **kwargs):
        super().__init__(**kwargs)
        self.ebno_db_min = ebno_db_min
        self.ebno_db_max = ebno_db_max
    def call(self, x, ebno_db=None, training=False, ebno_db_override=None):
        if ebno_db_override is not None:
            ebno_db = ebno_db_override
        if ebno_db is None:
            ebno_db = _tf.random.uniform([], self.ebno_db_min, self.ebno_db_max)
        noise_std = _tf.sqrt(_tf.reduce_mean(_tf.square(x)) /
                             (2.0 * 10.0 ** (float(ebno_db) / 10.0)) + 1e-10)
        return x + _tf.random.normal(_tf.shape(x), stddev=noise_std)

_channels_stub = types.ModuleType('models.channels')
_channels_stub.ChannelAWGN = _ChannelAWGN
_channels_stub.ChannelCDL  = _ChannelCDL
sys.modules['models.channels'] = _channels_stub
_models_stub.channels = _channels_stub

SCENE_CONFIGS = {
    "munich": {
        "source":      "builtin",
        "scene_ref":   srt.scene.munich,
        "bs_position": [8.5, 21.0, 27.0],
        "bs_look_at":  [45., 90., 1.5],
        "cam_position":[0, 80, 500],
        "cam_orient":  np.array([0., np.pi/2., -np.pi/2.]),
        "min_dist":    10,
        "max_dist":    400,
        "description": "Munich, Germany (Sionna built-in)",
    },
    "etoile": {
        "source":      "builtin",
        "scene_ref":   srt.scene.etoile,
        "bs_position": [0.0, 0.0, 30.0],
        "bs_look_at":  [100., 0., 1.5],
        "cam_position":[0, 0, 500],
        "cam_orient":  np.array([0., np.pi/2., 0.]),
        "min_dist":    10,
        "max_dist":    300,
        "description": "Paris, France — Place de l'Etoile (Sionna built-in)",
    },
    "sydney": {
        "source":      "custom",
        "scene_ref":   "/tmp/sydney_scene/sydney_cbd.xml",
        "extract_dir": "/tmp/sydney_scene",
        "bs_position": [-1.0, 15.7, 84.8],
        "bs_look_at":  [-1.0, 15.7, 1.5],
        "cam_position":[0, 0, 800],
        "cam_orient":  np.array([0., np.pi/2., 0.]),
        "min_dist":    10,
        "max_dist":    350,
        "description": "Sydney CBD, Australia (QVB area, custom OSM scene)",
    },
}

# 5G NR 系统参数
SUBCARRIER_SPACING = 30e3
NUM_TIME_STEPS     = 14
NUM_TX_ANT         = 4
NUM_RX_ANT         = 16
BATCH_SIZE_CIR     = 100
TX_POWER_DBM       = -10.0
NOISE_POWER_DBM    = -95.0
LATENT_DIM         = 512


def load_weights_correct(model, weights_path):
    def sort_key(path):
        parts = path.split('/')
        layer = parts[0]
        wtype = parts[-1].replace(':0', '')
        m     = re.search(r'_(\d+)$', layer)
        idx   = int(m.group(1)) if m else 0
        ltype = re.sub(r'_\d+$', '', layer)
        return (ltype, idx, wtype)

    h5_sorted = {}
    with h5py.File(weights_path, 'r') as f:
        for group in ['encoder', 'decoder']:
            if group not in f:
                continue
            entries = []
            def collect(name, obj, e=entries):
                if isinstance(obj, h5py.Dataset):
                    e.append((sort_key(name), np.array(obj)))
            f[group].visititems(collect)
            entries.sort(key=lambda x: x[0])
            h5_sorted[group] = [arr for _, arr in entries]

    total_loaded = 0
    for group in ['encoder', 'decoder']:
        submodel = getattr(model, group, None)
        if submodel is None or group not in h5_sorted:
            continue
        keras_vars = sorted(
            [(sort_key(v.path), v) for v in submodel.weights],
            key=lambda x: x[0]
        )
        for (_, var), arr in zip(keras_vars, h5_sorted[group]):
            if var.shape == arr.shape:
                var.assign(arr)
                total_loaded += 1

    print(f"权重加载：{total_loaded} / "
          f"{sum(len(v) for v in h5_sorted.values())} 个变量")
    return total_loaded


def build_model_and_lookup():
    from train_reconstruction import ReconstructionModel, psnr as semantic_psnr

    weight_files = sorted(glob.glob(
        os.path.join(WORK_DIR, f'checkpoints/image-jscc/recon_ld{LATENT_DIM}_*/best_psnr*.weights.h5')))
    if not weight_files:
        raise FileNotFoundError(f"找不到 ld={LATENT_DIM} 权重文件")
    weights_path = weight_files[-1]
    print(f"权重路径: {weights_path}")

    dummy     = tf.zeros((2, 32, 32, 3))
    sem_model = ReconstructionModel(latent_dim=LATENT_DIM, bypass=False, channel_type='analog')
    sem_model(dummy, training=False)
    load_weights_correct(sem_model, weights_path)

    (_, _), (x_test_raw, _) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test_raw[:200].astype('float32') / 255.0

    def evaluate_psnr_at_snr(snr_db):
        m = tf.keras.metrics.Mean()
        for batch in tf.data.Dataset.from_tensor_slices(x_test).batch(32):
            x_hat = tf.clip_by_value(
                sem_model(batch, ebno_db=float(snr_db), training=False), 0, 1)
            m.update_state(semantic_psnr(batch, x_hat))
        return float(m.result())

    print("预计算 PSNR-SNR 查找表...")
    lookup_snrs  = list(range(-10, 31, 2))
    lookup_psnrs = [evaluate_psnr_at_snr(s) for s in lookup_snrs]
    for s, p in zip(lookup_snrs, lookup_psnrs):
        print(f"  SNR={s:4d}dB → PSNR={p:.2f}dB")

    def snr_to_psnr(snr_db):
        return float(np.interp(snr_db, lookup_snrs, lookup_psnrs))

    return sem_model, x_test, lookup_snrs, lookup_psnrs, snr_to_psnr


def setup_scene(active_scene, sydney_zip=None):
    cfg = SCENE_CONFIGS[active_scene]
    if cfg["source"] == "builtin":
        scene = load_scene(cfg["scene_ref"])
    else:
        if not os.path.exists(cfg["extract_dir"]):
            if not sydney_zip or not os.path.exists(sydney_zip):
                raise FileNotFoundError(
                    f"Sydney 场景需要 zip 文件，请用 --sydney-zip 指定路径")
            print(f"解压 Sydney 场景: {sydney_zip}")
            with zipfile.ZipFile(sydney_zip, 'r') as z:
                z.extractall("/tmp/")
        scene = load_scene(cfg["scene_ref"])

    scene.frequency = 3.5e9
    scene.tx_array  = PlanarArray(
        num_rows=1, num_cols=NUM_RX_ANT // 2,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="tr38901", polarization="cross"
    )
    tx = Transmitter(name="tx", position=cfg["bs_position"],
                     look_at=cfg["bs_look_at"], display_radius=3.)
    scene.add(tx)
    return scene, cfg


def compute_or_load_cir(scene, cfg, active_scene):
    cache_a   = os.path.join(CACHE_DIR, f"{active_scene}_a.npy")
    cache_tau = os.path.join(CACHE_DIR, f"{active_scene}_tau.npy")
    cache_pos = os.path.join(CACHE_DIR, f"{active_scene}_ue_pos.npy")

    if os.path.exists(cache_a):
        a   = np.load(cache_a)
        tau = np.load(cache_tau)
        ue_positions_valid = np.load(cache_pos)
        print(f"从本地缓存加载 CIR（{active_scene}），有效 UE: {a.shape[0]}")
        return a, tau, ue_positions_valid

    print("计算 Radio Map（约 1-3 分钟）...")
    rm_solver = RadioMapSolver()
    rm = rm_solver(scene, max_depth=5, cell_size=(5., 5.), samples_per_tx=int(5e6))
    print("Radio Map 完成")

    scene.rx_array = PlanarArray(
        num_rows=1, num_cols=NUM_TX_ANT // 2,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="iso", polarization="cross"
    )

    # 先添加接收器
    first_pos, _ = rm.sample_positions(
        num_pos=BATCH_SIZE_CIR, metric="path_gain",
        min_val_db=-130, max_val_db=0,
        min_dist=cfg["min_dist"], max_dist=cfg["max_dist"], seed=0
    )
    for i in range(BATCH_SIZE_CIR):
        if scene.get(f"rx-{i}"):
            scene.remove(f"rx-{i}")
        scene.add(Receiver(name=f"rx-{i}",
                           position=first_pos[0][i].numpy(),
                           display_radius=1., color=(1, 0, 0)))

    target_num_cirs = 2000
    num_runs        = int(np.ceil(target_num_cirs / BATCH_SIZE_CIR))
    a_list, tau_list, ue_pos_list = [], [], []
    max_num_paths = 0
    p_solver = PathSolver()

    print(f"批量 CIR 生成（{num_runs} 批 × {BATCH_SIZE_CIR} 个位置）...")
    for run_idx in range(num_runs):
        print(f"  进度: {run_idx+1}/{num_runs}", end="\r")
        ue_pos, _ = rm.sample_positions(
            num_pos=BATCH_SIZE_CIR, metric="path_gain",
            min_val_db=-130, max_val_db=0,
            min_dist=cfg["min_dist"], max_dist=cfg["max_dist"],
            seed=run_idx
        )
        for i in range(BATCH_SIZE_CIR):
            scene.receivers[f"rx-{i}"].position = ue_pos[0][i].numpy()

        paths      = p_solver(scene, max_depth=5, max_num_paths_per_src=10**6)
        a_b, tau_b = paths.cir(
            sampling_frequency=SUBCARRIER_SPACING,
            num_time_steps=NUM_TIME_STEPS, out_type='numpy'
        )
        a_list.append(a_b)
        tau_list.append(tau_b)
        ue_pos_list.append(ue_pos[0].numpy())
        max_num_paths = max(max_num_paths, a_b.shape[-2])

    print()
    a_padded, tau_padded = [], []
    for a_, tau_ in zip(a_list, tau_list):
        pad_a   = [[0,0]] * (a_.ndim - 2) + \
                  [[0, max_num_paths - a_.shape[-2]]] + [[0,0]]
        pad_tau = [[0,0]] * (tau_.ndim - 1) + \
                  [[0, max_num_paths - tau_.shape[-1]]]
        a_padded.append(np.pad(a_, pad_a))
        tau_padded.append(np.pad(tau_, pad_tau))

    a   = np.concatenate(a_padded,   axis=0)
    tau = np.concatenate(tau_padded, axis=0)
    ue_positions_all = np.concatenate(ue_pos_list, axis=0)

    a   = np.transpose(a,   (0, 2, 3, 1, 4, 5))
    tau = np.transpose(tau, (0, 2, 1))

    valid = np.sum(np.abs(a)**2, axis=tuple(range(1, a.ndim))) > 0.
    a, tau = a[valid], tau[valid]
    ue_positions_valid = ue_positions_all[valid]

    np.save(cache_a,   a)
    np.save(cache_tau, tau)
    np.save(cache_pos, ue_positions_valid)
    print(f"CIR 已缓存到本地，共 {a.shape[0]} 个有效位置")
    return a, tau, ue_positions_valid


def compute_snr_psnr(a, ue_positions_valid, cfg, snr_to_psnr):
    bs_pos      = np.array(cfg["bs_position"])
    norm_factor = NUM_RX_ANT * NUM_TX_ANT
    snr_list, distance_list, psnr_list = [], [], []

    for idx in range(len(a)):
        path_gain = np.sum(np.abs(a[idx])**2) / norm_factor
        snr_db    = float(np.clip(
            TX_POWER_DBM + 10.0 * np.log10(max(path_gain, 1e-30)) - NOISE_POWER_DBM,
            -10.0, 30.0))
        dist = float(np.linalg.norm(ue_positions_valid[idx] - bs_pos))
        snr_list.append(snr_db)
        distance_list.append(dist)
        psnr_list.append(snr_to_psnr(snr_db))

    return (np.array(snr_list), np.array(distance_list), np.array(psnr_list))


def save_results(active_scene, ue_positions_valid, snr_arr, distance_arr, psnr_arr):
    out_json = os.path.join(EVAL_DIR, f"site_specific_results_{active_scene}.json")
    with open(out_json, 'w') as f:
        json.dump([{"idx": i,
                    "position": ue_positions_valid[i].tolist(),
                    "distance_m": round(float(distance_arr[i]), 1),
                    "snr_db":    round(float(snr_arr[i]),    2),
                    "psnr_db":   round(float(psnr_arr[i]),   2)}
                   for i in range(len(snr_arr))], f, indent=2)
    print(f"JSON 保存: {out_json}")
    return out_json


def plot_results(active_scene, cfg, snr_arr, distance_arr, psnr_arr,
                 ue_positions_valid, lookup_snrs, lookup_psnrs):
    xs, ys  = ue_positions_valid[:, 0], ue_positions_valid[:, 1]
    bs_pos  = np.array(cfg["bs_position"])
    norm_p  = mcolors.Normalize(vmin=psnr_arr.min(), vmax=psnr_arr.max())
    norm_s  = mcolors.Normalize(vmin=snr_arr.min(),  vmax=snr_arr.max())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Site-Specific Semantic Image Communication — Sionna RT\n"
        f"{cfg['description']} | CNN JSCC (ld={LATENT_DIM}) | "
        f"3.5 GHz | {len(snr_arr)} UE positions",
        fontsize=13, fontweight='bold'
    )

    ax = axes[0, 0]
    sc = ax.scatter(xs, ys, c=psnr_arr, cmap=cm.RdYlGn, norm=norm_p,
                    s=20, alpha=0.7, edgecolors='none')
    ax.scatter([bs_pos[0]], [bs_pos[1]], c='blue', marker='*', s=400, zorder=10,
               label='Base Station', edgecolors='navy', linewidths=0.8)
    plt.colorbar(sc, ax=ax, label='PSNR (dB)', pad=0.02)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title('(A) Top-View: UE Positions Colored by Semantic PSNR')
    ax.legend(fontsize=9); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    sc2 = ax.scatter(distance_arr, psnr_arr, c=snr_arr, cmap=cm.plasma_r,
                     norm=norm_s, s=15, alpha=0.6, edgecolors='none')
    plt.colorbar(sc2, ax=ax, label='RT SNR (dB)', pad=0.02)
    coeffs = np.polyfit(distance_arr, psnr_arr, 1)
    x_fit  = np.linspace(distance_arr.min(), distance_arr.max(), 200)
    ax.plot(x_fit, np.polyval(coeffs, x_fit), 'k--', lw=1.5, alpha=0.7,
            label=f'Trend ({coeffs[0]:.3f} dB/m)')
    ax.set_xlabel('Distance to BS (m)'); ax.set_ylabel('PSNR (dB)')
    ax.set_title('(B) Distance vs Semantic PSNR')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.scatter(snr_arr, psnr_arr, c='steelblue', s=15, alpha=0.5,
               label=f'RT site-specific (ld={LATENT_DIM})', zorder=3)
    for fpath, label, color, ls, marker in [
        (os.path.join(EVAL_DIR, "snr_recon_ld512.json"),     "AWGN ld=512",  "limegreen", "-",  "o"),
        (os.path.join(EVAL_DIR, "snr_recon_cdl_ld512.json"), "CDL-A ld=512", "orange",    "--", "s"),
        (os.path.join(EVAL_DIR, "jpeg_baseline.json"),       "JPEG+LDPC",    "red",       ":",  "D"),
    ]:
        if os.path.exists(fpath):
            with open(fpath) as f:
                ref = json.load(f)
            ax.plot([r['ebno_db'] for r in ref], [r['psnr'] for r in ref],
                    color=color, linestyle=ls, lw=2, marker=marker,
                    markersize=5, label=label, zorder=5)
    ax.plot(lookup_snrs, lookup_psnrs, 'k-', lw=1, alpha=0.4,
            label='Lookup table (AWGN, ld=512)', zorder=2)
    ax.set_xlabel('SNR / Eb/N0 (dB)'); ax.set_ylabel('PSNR (dB)')
    ax.set_title('(C) SNR vs PSNR: RT site-specific vs Reference Curves')
    ax.legend(fontsize=8, loc='lower right'); ax.grid(True, alpha=0.3)
    ax.set_xlim([-12, 32])

    ax = axes[1, 1]
    near = psnr_arr[distance_arr < 100]
    mid  = psnr_arr[(distance_arr >= 100) & (distance_arr < 250)]
    far  = psnr_arr[distance_arr >= 250]
    bp   = ax.boxplot(
        [near, mid, far],
        labels=[f'Near\n(<100m)\nn={len(near)}',
                f'Mid\n(100-250m)\nn={len(mid)}',
                f'Far\n(>250m)\nn={len(far)}'],
        patch_artist=True, medianprops=dict(color='black', lw=2)
    )
    for patch, color in zip(bp['boxes'], ['#2ecc71', '#f39c12', '#e74c3c']):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    for i, g in enumerate([near, mid, far], 1):
        if len(g) > 0:
            ax.scatter(i, g.mean(), color='black', marker='D', s=50, zorder=5)
            ax.text(i, g.mean() + 0.3, f'{g.mean():.1f}', ha='center', fontsize=8)
    if len(near) > 0 and len(far) > 0:
        ax.text(0.5, 0.95,
                f'Near vs Far: Δ={near.mean()-far.mean():.1f} dB',
                transform=ax.transAxes, ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('(D) PSNR Distribution by Distance Group')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_png = os.path.join(EVAL_DIR, f"site_specific_results_{active_scene}.png")
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表保存: {out_png}")

    # 热力图
    grid_resolution = 200
    x_min, x_max = xs.min() - 20, xs.max() + 20
    y_min, y_max = ys.min() - 20, ys.max() + 20
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution)
    )
    grid_psnr = griddata(np.column_stack([xs, ys]), psnr_arr,
                         (grid_x, grid_y), method='linear')
    norm_h   = mcolors.Normalize(vmin=psnr_arr.min(), vmax=psnr_arr.max())
    rgba_img = cm.RdYlGn(norm_h(np.nan_to_num(grid_psnr, nan=0.0)))
    rgba_img[..., 3] = np.where(np.isnan(grid_psnr), 0.0, 0.85)

    fig2, ax2 = plt.subplots(figsize=(10, 9))
    ax2.imshow(rgba_img, extent=[x_min, x_max, y_min, y_max],
               origin='lower', aspect='equal', interpolation='bilinear')
    sm = cm.ScalarMappable(cmap=cm.RdYlGn, norm=norm_h)
    sm.set_array([])
    plt.colorbar(sm, ax=ax2, label='Semantic PSNR (dB)', pad=0.02, shrink=0.8)
    ax2.scatter(xs, ys, c='white', s=8, alpha=0.3, edgecolors='none')
    ax2.scatter([bs_pos[0]], [bs_pos[1]], c='blue', marker='*', s=500, zorder=10,
                label='Base Station', edgecolors='navy', linewidths=0.8)
    ax2.set_xlabel('X (m)', fontsize=12); ax2.set_ylabel('Y (m)', fontsize=12)
    ax2.set_title(
        f'PSNR Heatmap — {cfg["description"]}\n'
        f'CNN JSCC (ld={LATENT_DIM}) | 3.5 GHz | {len(snr_arr)} UE positions',
        fontsize=13, fontweight='bold'
    )
    ax2.legend(fontsize=10); ax2.grid(True, alpha=0.2, color='gray')
    plt.tight_layout()
    out_heatmap = os.path.join(EVAL_DIR, f"site_specific_heatmap_{active_scene}.png")
    plt.savefig(out_heatmap, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"热力图保存: {out_heatmap}")


def print_summary(active_scene, cfg, snr_arr, distance_arr, psnr_arr):
    near_mask = distance_arr < 100
    mid_mask  = (distance_arr >= 100) & (distance_arr < 250)
    far_mask  = distance_arr >= 250
    print(f"""
{'='*60}
SITE-SPECIFIC EVALUATION SUMMARY
{'='*60}
场景：    {cfg['description']}
频率：    3.5 GHz
基站：    {cfg['bs_position']}，{NUM_RX_ANT} 天线
有效位置：{len(snr_arr)} 个

SNR：   {snr_arr.min():.1f} ~ {snr_arr.max():.1f} dB（均值 {snr_arr.mean():.1f} dB）
PSNR：  {psnr_arr.min():.2f} ~ {psnr_arr.max():.2f} dB（均值 {psnr_arr.mean():.2f} dB）

近距离 (<100m)，   n={near_mask.sum():3d}：PSNR = {psnr_arr[near_mask].mean():.2f} dB
中距离 (100-250m)，n={mid_mask.sum():3d}：PSNR = {psnr_arr[mid_mask].mean():.2f} dB
远距离 (>250m)，   n={far_mask.sum():3d}：PSNR = {psnr_arr[far_mask].mean():.2f} dB

★ 近距 vs 远距 PSNR 差 = {psnr_arr[near_mask].mean() - psnr_arr[far_mask].mean():.2f} dB
{'='*60}
""")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default='munich',
                        choices=['munich', 'etoile', 'sydney'],
                        help='场景选择（默认 munich）')
    parser.add_argument('--sydney-zip', default=None,
                        help='Sydney 场景 zip 文件路径（仅 --scene sydney 时需要）')
    parser.add_argument('--force-recompute', action='store_true',
                        help='忽略本地 CIR 缓存，强制重新计算')
    args = parser.parse_args()

    if args.force_recompute:
        for key in ['a', 'tau', 'ue_pos']:
            p = os.path.join(CACHE_DIR, f"{args.scene}_{key}.npy")
            if os.path.exists(p):
                os.remove(p)
                print(f"已删除缓存: {p}")

    print(f"\n=== 场景: {args.scene} ===")
    _, _, lookup_snrs, lookup_psnrs, snr_to_psnr = build_model_and_lookup()
    scene, cfg = setup_scene(args.scene, args.sydney_zip)
    a, tau, ue_positions_valid = compute_or_load_cir(scene, cfg, args.scene)
    snr_arr, distance_arr, psnr_arr = compute_snr_psnr(
        a, ue_positions_valid, cfg, snr_to_psnr)
    save_results(args.scene, ue_positions_valid, snr_arr, distance_arr, psnr_arr)
    plot_results(args.scene, cfg, snr_arr, distance_arr, psnr_arr,
                 ue_positions_valid, lookup_snrs, lookup_psnrs)
    print_summary(args.scene, cfg, snr_arr, distance_arr, psnr_arr)
