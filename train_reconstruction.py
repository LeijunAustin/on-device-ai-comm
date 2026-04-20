import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# 必须在 TF 初始化之前设置
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

"""
Image Reconstruction Semantic Communication
CNN encoder + AWGN channel + CNN decoder
Loss: MSE + SSIM
Metrics: PSNR / SSIM vs SNR
"""
import os, json, argparse
from models.channels import ChannelAWGN, ChannelCDL
import numpy as np
import tensorflow as tf
from datetime import datetime


# ─── 数据加载 ───────────────────────────────────────────
def setup_gpu():
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def augment(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.pad_to_bounding_box(image, 4, 4, 40, 40)
    image = tf.image.random_crop(image, (32, 32, 3))
    return image

def load_cifar10(batch_size=64, augment_train=True):
    (x_tr, _), (x_te, _) = tf.keras.datasets.cifar10.load_data()
    x_tr = x_tr.astype('float32') / 255.0
    x_te = x_te.astype('float32') / 255.0

    def aug_fn(x): return augment(x)

    train_ds = tf.data.Dataset.from_tensor_slices(x_tr)
    if augment_train:
        train_ds = train_ds.shuffle(10000).map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_ds = (tf.data.Dataset.from_tensor_slices(x_te)
               .batch(batch_size).prefetch(tf.data.AUTOTUNE))

    print(f"Train: {len(x_tr)}, Test: {len(x_te)}")
    return train_ds, test_ds


# ─── 模型定义 ───────────────────────────────────────────
def residual_block(x, filters, stride=1):
    shortcut = x
    x = tf.keras.layers.Conv2D(filters, 3, strides=stride, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    x = tf.keras.layers.Add()([x, shortcut])
    return tf.keras.layers.ReLU()(x)


def build_encoder(latent_dim=128):
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(latent_dim)(x)
    return tf.keras.Model(inputs, x, name='encoder')


def build_decoder(latent_dim=128):
    inputs = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(8 * 8 * 256, activation='relu')(inputs)
    x = tf.keras.layers.Reshape((8, 8, 256))(x)
    x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2DTranspose(3, 3, padding='same', activation='sigmoid')(x)
    return tf.keras.Model(inputs, x, name='decoder')


class AnalogChannel(tf.keras.layers.Layer):
    """Normalize + add AWGN noise."""
    def __init__(self, ebno_db_min=0, ebno_db_max=20, **kwargs):
        super().__init__(**kwargs)
        self.ebno_db_min = ebno_db_min
        self.ebno_db_max = ebno_db_max

    def call(self, z, ebno_db=None, training=False):
        # Power normalization
        pwr = tf.reduce_mean(tf.square(z), keepdims=True)
        z_norm = z / tf.sqrt(pwr + 1e-8)

        if ebno_db is not None:
            snr = tf.pow(10.0, tf.cast(ebno_db, tf.float32) / 10.0)
        elif training:
            ebno_db_rand = tf.random.uniform((), self.ebno_db_min, self.ebno_db_max)
            snr = tf.pow(10.0, ebno_db_rand / 10.0)
        else:
            snr = tf.pow(10.0, tf.cast(self.ebno_db_max, tf.float32) / 10.0)

        noise_std = 1.0 / tf.sqrt(snr)
        noise = tf.random.normal(tf.shape(z_norm), stddev=noise_std)
        return z_norm + noise


class ReconstructionModel(tf.keras.Model):
    def __init__(self, latent_dim=128, ebno_db_min=0, ebno_db_max=20,
                 bypass=False, channel_type='analog',
                 cdl_model='A', fec_type='LDPC5G', fec_num_iter=6,
                 channel_num_tx_ant=2, channel_num_rx_ant=2,
                 num_bits_per_symbol=4):
        super().__init__()
        self.encoder = build_encoder(latent_dim)
        self.bypass = bypass
        self.channel_type = channel_type

        if channel_type == 'analog':
            self.channel = AnalogChannel(ebno_db_min, ebno_db_max)
        elif channel_type == 'CDL':
            self.channel = ChannelCDL(
                fec_type=fec_type,
                cdl_model=cdl_model,
                channel_num_tx_ant=channel_num_tx_ant,
                channel_num_rx_ant=channel_num_rx_ant,
                num_bits_per_symbol=num_bits_per_symbol,
                fec_num_iter=fec_num_iter,
                ebno_db_min=ebno_db_min,
                ebno_db_max=ebno_db_max,
            )
        self.decoder = build_decoder(latent_dim)

    def call(self, x, ebno_db=None, training=False):
        z = self.encoder(x, training=training)
        if self.bypass:
            z_hat = z
        elif self.channel_type == 'analog':
            z_hat = self.channel(z, ebno_db=ebno_db, training=training)
        elif self.channel_type == 'CDL':
            # CDL: z -> flatten to bits -> channel -> reshape back
            z_shape = tf.shape(z)
            z_flat = tf.reshape(z, [-1])
            # binarize: sign-based
            z_bits = tf.cast(z_flat > 0, tf.float32)
            z_rx = self.channel(z_bits, ebno_db_override=ebno_db)
            # restore shape: map bits back to ±1
            z_hat = tf.reshape(2.0 * z_rx - 1.0, z_shape)
        x_hat = self.decoder(z_hat, training=training)
        return x_hat


# ─── 感知损失（VGG16 特征层） ──────────────────────────────
_perceptual_model = None

def get_perceptual_model():
    global _perceptual_model
    if _perceptual_model is None:
        vgg = tf.keras.applications.VGG16(
            include_top=False, weights='imagenet', input_shape=(32, 32, 3))
        vgg.trainable = False
        # block2_conv2：对 32×32 输入仍有 8×8 的空间分辨率，信息量合适
        out = vgg.get_layer('block2_conv2').output
        _perceptual_model = tf.keras.Model(vgg.input, out, name='perceptual_vgg')
        _perceptual_model.trainable = False
    return _perceptual_model

def perceptual_loss_fn(y_true, y_pred):
    prep = tf.keras.applications.vgg16.preprocess_input
    feat_t = get_perceptual_model()(prep(y_true * 255.0), training=False)
    feat_p = get_perceptual_model()(prep(y_pred * 255.0), training=False)
    # 归一化后计算 L2 距离，避免量级差异
    norm = tf.cast(tf.size(feat_t), tf.float32)
    return tf.reduce_sum(tf.square(feat_t - feat_p)) / norm


# ─── 损失函数 ───────────────────────────────────────────
def ssim_loss(y_true, y_pred):
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def reconstruction_loss(y_true, y_pred, alpha=0.8, perceptual_weight=0.0):
    mse  = tf.reduce_mean(tf.square(y_true - y_pred))
    ssim = ssim_loss(y_true, y_pred)
    loss = alpha * mse + (1 - alpha) * ssim
    if perceptual_weight > 0:
        loss = loss + perceptual_weight * perceptual_loss_fn(y_true, y_pred)
    return loss

def psnr(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))

def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


# ─── 训练 ───────────────────────────────────────────────
def train(args):
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    suffix = '_perceptual' if args.perceptual_weight > 0 else ''
    out = os.path.join(args.output_dir, f'recon_ld{args.latent_dim}{suffix}_{ts}')
    os.makedirs(out, exist_ok=True)
    print(f"Output: {out}")

    train_ds, test_ds = load_cifar10(args.batch_size)

    model = ReconstructionModel(
        latent_dim=args.latent_dim,
        ebno_db_min=args.ebno_db_min,
        ebno_db_max=args.ebno_db_max,
        bypass=args.bypass,
        channel_type=args.channel_type,
        cdl_model=args.cdl_model,
        fec_type=args.fec_type,
        fec_num_iter=args.fec_num_iter,
        channel_num_tx_ant=args.tx_ant,
        channel_num_rx_ant=args.rx_ant,
        num_bits_per_symbol=args.bits_per_sym,
    )

    # 感知损失微调模式：从已有权重继续训练
    if args.finetune_from:
        dummy = tf.zeros((2, 32, 32, 3))
        model(dummy, training=False)
        model.load_weights(args.finetune_from)
        print(f"Loaded weights for fine-tuning: {args.finetune_from}")
        if args.perceptual_weight > 0:
            _ = get_perceptual_model()  # 预加载 VGG 权重

    opt = tf.keras.optimizers.Adam(args.lr)
    best_psnr = 0.0
    history = []

    for epoch in range(args.epochs):
        tr_loss = tf.keras.metrics.Mean()
        tr_psnr = tf.keras.metrics.Mean()

        for imgs in train_ds:
            with tf.GradientTape() as tape:
                x_hat = model(imgs, training=True)
                loss = reconstruction_loss(imgs, x_hat,
                                           perceptual_weight=args.perceptual_weight)
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            tr_loss.update_state(loss)
            tr_psnr.update_state(psnr(imgs, x_hat))

        # eval
        te_psnr = tf.keras.metrics.Mean()
        te_ssim = tf.keras.metrics.Mean()
        for imgs in test_ds:
            x_hat = model(imgs, training=False)
            te_psnr.update_state(psnr(imgs, x_hat))
            te_ssim.update_state(ssim_metric(imgs, x_hat))

        res = {
            'epoch': epoch + 1,
            'train_loss': float(tr_loss.result()),
            'train_psnr': float(tr_psnr.result()),
            'test_psnr':  float(te_psnr.result()),
            'test_ssim':  float(te_ssim.result()),
        }
        history.append(res)
        print(f"Ep{epoch+1:3d}/{args.epochs}  "
              f"Loss:{res['train_loss']:.4f}  "
              f"TrainPSNR:{res['train_psnr']:.2f}  "
              f"TestPSNR:{res['test_psnr']:.2f}  "
              f"SSIM:{res['test_ssim']:.4f}")

        if res['test_psnr'] > best_psnr:
            best_psnr = res['test_psnr']
            model.save_weights(os.path.join(out, f'best_psnr{best_psnr:.2f}.weights.h5'))
            print(f"  ✅ Best PSNR: {best_psnr:.2f} dB")

        with open(os.path.join(out, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    print(f"\nDone. Best PSNR: {best_psnr:.2f} dB → {out}")
    return out, model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--latent-dim',   type=int,   default=128)
    p.add_argument('--ebno-db-min',  type=float, default=0.0)
    p.add_argument('--ebno-db-max',  type=float, default=20.0)
    p.add_argument('--bypass',       action='store_true')
    p.add_argument('--batch-size',   type=int,   default=64)
    p.add_argument('--epochs',       type=int,   default=50)
    p.add_argument('--lr',           type=float, default=1e-3)
    p.add_argument('--output-dir',   default='checkpoints/image-jscc')
    p.add_argument('--channel-type',  default='analog', choices=['analog','CDL'])
    p.add_argument('--cdl-model',     default='A')
    p.add_argument('--fec-type',      default='LDPC5G')
    p.add_argument('--fec-num-iter',  type=int, default=6)
    p.add_argument('--tx-ant',        type=int, default=2)
    p.add_argument('--rx-ant',        type=int, default=2)
    p.add_argument('--bits-per-sym',      type=int,   default=4)
    p.add_argument('--perceptual-weight', type=float, default=0.0,
                   help='VGG perceptual loss weight (0=disabled, 0.1 recommended for finetune)')
    p.add_argument('--finetune-from',     default=None,
                   help='Load existing weights and fine-tune (path to .weights.h5)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print("Config:", vars(args))
    train(args)
