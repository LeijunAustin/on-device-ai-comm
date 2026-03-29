"""
Image Semantic Communication Model
Replaces BART encoder/decoder with image semantic encoder/decoder
Keeps VQ + Channel pipeline from original project unchanged

Architecture:
  ImageEncoder → LatentAdapter → VQ → BitMapper → Channel → Dequantizer → LatentAdapter → TaskDecoder
"""
import tensorflow as tf
import numpy as np
from .vq_vae import VectorQuantizer
from .channels import ChannelAWGN, ChannelCDL
from .utils import tensor_to_binary_v2, binary_to_tensor_v2, replace_nan, get_ber


# ─────────────────────────────────────────────────────────
# 1. CNN Image Encoder (Stage 1 baseline)
# ─────────────────────────────────────────────────────────
def residual_block(x, filters, stride=1):
    """ResNet-style residual block."""
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
    x = tf.keras.layers.ReLU()(x)
    return x


class CNNImageEncoder(tf.keras.layers.Layer):
    """
    ResNet-style CNN image semantic encoder.
    Input : (B, H, W, 3)  e.g. CIFAR-10: (B, 32, 32, 3)
    Output: (B, seq_len, embedding_dim)  -- compatible with VQ layer
    """
    def __init__(self, seq_len=32, embedding_dim=2, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.adapter = tf.keras.layers.Dense(seq_len * embedding_dim)

        # Build ResNet backbone as functional model
        inputs = tf.keras.Input(shape=(32, 32, 3))
        x = tf.keras.layers.Conv2D(64, 3, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = residual_block(x, 64)
        x = residual_block(x, 64)
        x = residual_block(x, 128, stride=2)
        x = residual_block(x, 128)
        x = residual_block(x, 256, stride=2)
        x = residual_block(x, 256)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        self.backbone = tf.keras.Model(inputs, x, name='resnet_backbone')

    def call(self, x, training=False):
        x = self.backbone(x, training=training)
        x = self.adapter(x)
        return tf.reshape(x, (-1, self.seq_len, self.embedding_dim))


# ─────────────────────────────────────────────────────────
# 2. SegFormer-B0 Image Encoder (Stage 2 main model)
#    Lightweight ViT designed for semantic understanding
# ─────────────────────────────────────────────────────────
class MixFFN(tf.keras.layers.Layer):
    """Mix-FFN used in SegFormer."""
    def __init__(self, in_features, hidden_features, **kwargs):
        super().__init__(**kwargs)
        self.fc1   = tf.keras.layers.Dense(hidden_features)
        self.dw    = tf.keras.layers.DepthwiseConv2D(3, padding='same', activation='gelu')
        self.fc2   = tf.keras.layers.Dense(in_features)

    def call(self, x, H, W, training=False):
        x = self.fc1(x)
        B = tf.shape(x)[0]
        x_2d = tf.reshape(x, (B, H, W, -1))
        x_2d = self.dw(x_2d, training=training)
        x = tf.reshape(x_2d, (B, H * W, -1))
        x = self.fc2(x)
        return x


class EfficientSelfAttention(tf.keras.layers.Layer):
    """Efficient self-attention with sequence reduction (SR)."""
    def __init__(self, dim, num_heads, sr_ratio=1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.q  = tf.keras.layers.Dense(dim)
        self.kv = tf.keras.layers.Dense(dim * 2)
        self.proj = tf.keras.layers.Dense(dim)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr   = tf.keras.layers.Conv2D(dim, sr_ratio, strides=sr_ratio, padding='same')
            self.norm = tf.keras.layers.LayerNormalization()

    def call(self, x, H, W, training=False):
        B = tf.shape(x)[0]
        N = H * W
        q = self.q(x)
        q = tf.reshape(q, (B, N, self.num_heads, self.head_dim))
        q = tf.transpose(q, (0, 2, 1, 3))

        if self.sr_ratio > 1:
            x_ = tf.reshape(x, (B, H, W, -1))
            x_ = self.sr(x_, training=training)
            x_ = tf.reshape(x_, (B, -1, tf.shape(x_)[-1]))
            x_ = self.norm(x_)
        else:
            x_ = x

        kv = self.kv(x_)
        kv = tf.reshape(kv, (B, -1, 2, self.num_heads, self.head_dim))
        kv = tf.transpose(kv, (2, 0, 3, 1, 4))
        k, v = kv[0], kv[1]

        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        x = tf.matmul(attn, v)
        x = tf.transpose(x, (0, 2, 1, 3))
        x = tf.reshape(x, (B, N, -1))
        return self.proj(x)


class SegFormerBlock(tf.keras.layers.Layer):
    """One SegFormer transformer block."""
    def __init__(self, dim, num_heads, mlp_ratio=4, sr_ratio=1, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.attn  = EfficientSelfAttention(dim, num_heads, sr_ratio)
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.ffn   = MixFFN(dim, dim * mlp_ratio)

    def call(self, x, H, W, training=False):
        x = x + self.attn(self.norm1(x), H, W, training=training)
        x = x + self.ffn(self.norm2(x), H, W, training=training)
        return x


class OverlapPatchEmbed(tf.keras.layers.Layer):
    """Overlapping patch embedding for SegFormer."""
    def __init__(self, patch_size=7, stride=4, embed_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.proj = tf.keras.layers.Conv2D(
            embed_dim, patch_size, strides=stride, padding='same')
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, x, training=False):
        x = self.proj(x, training=training)
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x = tf.reshape(x, (B, H * W, C))
        x = self.norm(x)
        return x, H, W


class SegFormerB0Encoder(tf.keras.layers.Layer):
    """
    SegFormer-B0 image semantic encoder (lightweight ViT).
    Input : (B, H, W, 3)
    Output: (B, seq_len, embedding_dim)
    """
    # B0 config: dims=[32,64,160,256], depths=[2,2,2,2], heads=[1,2,5,8]
    def __init__(self, seq_len=32, embedding_dim=2, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim

        dims   = [32, 64, 160, 256]
        depths = [2,  2,  2,   2]
        heads  = [1,  2,  5,   8]
        sr_ratios = [8, 4, 2,  1]

        # 4 stages of patch embedding + transformer blocks
        self.patch_embeds = []
        self.blocks       = []
        self.norms        = []

        strides     = [4, 2, 2, 2]
        patch_sizes = [7, 3, 3, 3]

        for i in range(4):
            self.patch_embeds.append(
                OverlapPatchEmbed(patch_sizes[i], strides[i], dims[i]))
            stage_blocks = [
                SegFormerBlock(dims[i], heads[i], sr_ratio=sr_ratios[i])
                for _ in range(depths[i])
            ]
            self.blocks.append(stage_blocks)
            self.norms.append(tf.keras.layers.LayerNormalization())

        # LatentAdapter: fuse 4-scale features → (B, seq_len, embedding_dim)
        self.adapter = tf.keras.layers.Dense(seq_len * embedding_dim)

    def call(self, x, training=False):
        B = tf.shape(x)[0]
        outs = []
        for i in range(4):
            x, H, W = self.patch_embeds[i](x, training=training)
            for block in self.blocks[i]:
                x = block(x, H, W, training=training)
            x = self.norms[i](x)
            # reshape back to spatial for next stage
            x_2d = tf.reshape(x, (B, H, W, -1))
            outs.append(tf.reduce_mean(x, axis=1))  # global average pool
            x = x_2d  # pass spatial map to next stage

        # fuse all 4 scales
        fused = tf.concat(outs, axis=-1)
        out   = self.adapter(fused)
        return tf.reshape(out, (-1, self.seq_len, self.embedding_dim))


# ─────────────────────────────────────────────────────────
# 3. Task Decoders
# ─────────────────────────────────────────────────────────
class ClassificationDecoder(tf.keras.layers.Layer):
    """
    For Stage 1: CIFAR-10 classification.
    Input : (B, seq_len, embedding_dim)
    Output: (B, num_classes)  logits
    """
    def __init__(self, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.fc2 = tf.keras.layers.Dense(num_classes)

    def call(self, x, training=False):
        x = self.flatten(x)
        x = self.fc1(x, training=training)
        x = self.dropout(x, training=training)
        return self.fc2(x)


class ImageReconstructionDecoder(tf.keras.layers.Layer):
    """
    For Stage 2: Image reconstruction.
    Input : (B, seq_len, embedding_dim)
    Output: (B, H, W, 3)
    """
    def __init__(self, output_shape=(32, 32, 3), **kwargs):
        super().__init__(**kwargs)
        self.img_shape = output_shape
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(4 * 4 * 256, activation='relu'),
            tf.keras.layers.Reshape((4, 4, 256)),
            tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(64,  3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(32,  3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2DTranspose(3,   3, padding='same', activation='sigmoid'),
        ])

    def call(self, x, training=False):
        x = tf.reshape(x, (tf.shape(x)[0], -1))
        return self.decoder(x, training=training)


# ─────────────────────────────────────────────────────────
# 4. Complete Image Semantic Communication Model
# ─────────────────────────────────────────────────────────
class ImageSemanticCommModel(tf.keras.Model):
    """
    Full pipeline:
      ImageEncoder → VQ → BitMapper → Channel → Dequantizer → TaskDecoder

    Args:
        encoder_type : 'cnn' or 'segformer'
        task         : 'classification' or 'reconstruction'
        channel_type : 'AWGN' or 'CDL'
    """
    def __init__(self,
                 encoder_type='cnn',
                 task='classification',
                 num_classes=10,
                 seq_len=32,
                 embedding_dim=2,
                 num_embeddings=256,
                 channel_type='CDL',
                 cdl_model='A',
                 channel_num_tx_ant=2,
                 channel_num_rx_ant=2,
                 num_bits_per_symbol=4,
                 fec_type='Polar5G',
                 fec_num_iter=20,
                 ebno_db=None,
                 ebno_db_min=5,
                 ebno_db_max=15,
                 do_train=True,
                 bypass_channel=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.do_train = do_train
        self.task = task
        self.bypass_channel = bypass_channel

        # 1. Image Encoder
        if encoder_type == 'cnn':
            self.image_encoder = CNNImageEncoder(seq_len, embedding_dim)
        elif encoder_type == 'segformer':
            self.image_encoder = SegFormerB0Encoder(seq_len, embedding_dim)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # 2. VQ layer (reuse from original project)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim,
                                        name="vector_quantizer")

        # 3. Channel (reuse from original project)
        ch_config = dict(
            fec_type=fec_type,
            cdl_model=cdl_model,
            channel_num_tx_ant=channel_num_tx_ant,
            channel_num_rx_ant=channel_num_rx_ant,
            num_bits_per_symbol=int(num_bits_per_symbol),
            ebno_db=ebno_db,
            ebno_db_min=ebno_db_min,
            ebno_db_max=ebno_db_max,
            fec_num_iter=fec_num_iter,
        )
        if channel_type == 'CDL':
            self.channel = ChannelCDL(**ch_config)
        elif channel_type == 'AWGN':
            self.channel = ChannelAWGN(
                fec_type=fec_type,
                num_bits_per_symbol=int(num_bits_per_symbol),
                fec_n=1024, fec_k=512,
                ebno_db=ebno_db,
                ebno_db_min=ebno_db_min,
                ebno_db_max=ebno_db_max,
                fec_num_iter=fec_num_iter,
            )

        # 4. Task Decoder
        if task == 'classification':
            self.task_decoder = ClassificationDecoder(num_classes)
        elif task == 'reconstruction':
            self.task_decoder = ImageReconstructionDecoder()
        else:
            raise ValueError(f"Unknown task: {task}")

        # VQ pipeline functions (same as original project)
        self._tensor_to_binary = [
            self.vq_layer.get_code_indices,
            tensor_to_binary_v2
        ]
        self._binary_to_tensor = [
            binary_to_tensor_v2,
            self.vq_layer.handle_invalid_values,
            self.vq_layer.reconstruct_with_indices
        ]

    def call(self, images, training=False):
        # Step 1: Image semantic encoding
        z = self.image_encoder(images, training=training)
        shape = tf.shape(z)

        if self.bypass_channel:
            # 直接跳过 VQ 和信道，验证编解码器本身能学习
            z_hat = z
            ber = tf.constant(-1.0)
        else:
            # Step 2: VQ with Straight-Through Estimator
            # z_ste: gradients flow through (for encoder training)
            # vq_layer.losses: commitment + codebook loss added automatically
            z_ste = self.vq_layer(z)  # STE output, gradients can flow

            # Step 3: VQ → binary (use z_ste for binarization)
            z_q = z_ste
            for f in self._tensor_to_binary:
                z_q = f(z_q)
            z_binary = z_q

            # Step 4: Channel
            _ebno = getattr(self, '_eval_ebno_db', None)
            z_rx = self.channel(z_binary, ebno_db_override=_ebno)

            # Step 5: Binary → latent
            z_hat = z_rx
            for f in self._binary_to_tensor:
                z_hat = f(z_hat)
            z_hat = tf.reshape(z_hat, shape)
            z_hat = replace_nan(z_hat, 0)

            # Step 6: BER (eval only)
            if not self.do_train:
                ber = get_ber(z_binary, tf.reshape(z_rx, tf.shape(z_binary)))
            else:
                ber = tf.constant(-1.0)

        # Step 7: Task decode
        output = self.task_decoder(z_hat, training=training)

        return output, ber


# ─────────────────────────────────────────────────────────
# 5. Factory function
# ─────────────────────────────────────────────────────────
def build_image_comm_model(encoder_type='cnn', task='classification', **kwargs):
    return ImageSemanticCommModel(
        encoder_type=encoder_type,
        task=task,
        **kwargs
    )
