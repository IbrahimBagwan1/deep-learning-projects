"""
Full, from-scratch Transformer ("Attention Is All You Need") implemented in TensorFlow 2.x
with a simple machine translation training pipeline using the TED Talks Portuguese->English dataset
(from tensorflow_datasets). This is a single-file script you can run directly.

Requirements:
  pip install tensorflow tensorflow-datasets

Run:
  python transformer_tensorflow_machine_translation.py

Notes:
 - This implementation uses tfds and SubwordTextEncoder for tokenization (builds BPE-like subword vocab).
 - For quick experiments it uses small model dims; for real training increase d_model, N, and training time.
"""

import os
import time
import math
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers

# -------------------------
# Hyperparameters
# -------------------------
BUFFER_SIZE = 20000
BATCH_SIZE = 64
MAX_LENGTH = 40  # max tokens per sentence (truncate longer)
EPOCHS = 20

# Model hyperparameters (small for quick experiments)
NUM_LAYERS = 4
D_MODEL = 128
D_FF = 512
NUM_HEADS = 8
DROPOUT_RATE = 0.1

# -------------------------
# Positional encoding
# -------------------------
def get_angles(pos, i, d_model):
    angle_rates = 1 / (10000 ** (2 * (i // 2) / float(d_model)))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(
        pos=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
        i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
        d_model=d_model)

    # apply sin to even indices in the array; cos to odd indices
    sines = tf.math.sin(angle_rads[:, 0::2])
    coses = tf.math.cos(angle_rads[:, 1::2])

    pos_encoding = tf.concat([sines, coses], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

# -------------------------
# Masking helpers
# -------------------------
def create_padding_mask(seq):
    # seq: (batch, seq_len)
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding
    # mask to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

# -------------------------
# Scaled dot-product attention
# -------------------------
def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_q, seq_k)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        # mask shape broadcastable to scaled_logits
        scaled_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_logits, axis=-1)
    output = tf.matmul(attention_weights, v)  # (..., seq_q, depth_v)
    return output, attention_weights

# -------------------------
# Multi-head attention layer
# -------------------------
class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)

        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        # x: (batch_size, seq_len, d_model)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch, heads, seq_len, depth)

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attn_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch, seq_len, heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output, attn_weights

# -------------------------
# Point-wise feed forward network
# -------------------------
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        layers.Dense(dff, activation='relu'),
        layers.Dense(d_model)
    ])

# -------------------------
# Encoder and Decoder layers
# -------------------------
class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

class DecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2

# -------------------------
# Encoder and Decoder stacks
# -------------------------
class Encoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)  # (batch, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch, input_seq_len, d_model)

class Decoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        return x, attention_weights

# -------------------------
# Full Transformer model
# -------------------------
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        super().__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)

        self.final_layer = layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch, inp_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)  # (batch, tar_seq_len, target_vocab_size)
        return final_output, attention_weights

# -------------------------
# Learning rate schedule (as in paper)
# -------------------------
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

# -------------------------
# Loss and metrics
# -------------------------
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# -------------------------
# Data preparation (tfds + SubwordTextEncoder)
# -------------------------
print("Loading dataset (this will download if not present)...")
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

# Build tokenizers from training data
print("Building subword tokenizers (may take a few minutes)...")
train_subwords_src = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy().decode('utf-8') for pt, en in train_examples), target_vocab_size=2**13)
train_subwords_tgt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy().decode('utf-8') for pt, en in train_examples), target_vocab_size=2**13)

INPUT_VOCAB_SIZE = train_subwords_src.vocab_size + 2  # for start and end tokens
TARGET_VOCAB_SIZE = train_subwords_tgt.vocab_size + 2

START_TOKEN, END_TOKEN = [train_subwords_src.vocab_size], [train_subwords_src.vocab_size + 1]
# Note: We'll use separate tokenizers; START/END tokens will be using respective vocab sizes when encoding.

# Helper encode
def encode(lang1, lang2):
    lang1 = train_subwords_src.encode(lang1.numpy().decode('utf-8'))
    lang2 = train_subwords_tgt.encode(lang2.numpy().decode('utf-8'))

    lang1 = [train_subwords_src.vocab_size] + lang1 + [train_subwords_src.vocab_size + 1]
    lang2 = [train_subwords_tgt.vocab_size] + lang2 + [train_subwords_tgt.vocab_size + 1]

    return lang1, lang2

def tf_encode(pt, en):
    result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
    result_pt.set_shape([None])
    result_en.set_shape([None])
    return result_pt, result_en

def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)

print("Preparing tokenized and batched datasets...")
train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(lambda x, y: filter_max_length(x, y))
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([None], [None]))
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

val_dataset = val_examples.map(tf_encode)
val_dataset = val_dataset.filter(lambda x, y: filter_max_length(x, y))
val_dataset = val_dataset.padded_batch(BATCH_SIZE, padded_shapes=([None], [None]))

# -------------------------
# Create model, optimizer, checkpointing
# -------------------------
print("Building Transformer model...")
transformer = Transformer(
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dff=D_FF,
    input_vocab_size=INPUT_VOCAB_SIZE,
    target_vocab_size=TARGET_VOCAB_SIZE,
    pe_input=1000,
    pe_target=1000,
    rate=DROPOUT_RATE)

learning_rate = CustomSchedule(D_MODEL)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

checkpoint_path = "./checkpoints/transformer"
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

# -------------------------
# Training step (tf.function)
# -------------------------
@tf.function
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar_inp)[1])
    dec_target_padding_mask = create_padding_mask(tar_inp)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, training=True, enc_padding_mask=enc_padding_mask, look_ahead_mask=combined_mask, dec_padding_mask=dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)

# -------------------------
# Training loop
# -------------------------
print("Starting training...")
for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_state()
    train_accuracy.reset_state()

    # Train
    for (batch, (inp, tar)) in enumerate(train_dataset):
        train_step(inp, tar)
        if batch % 100 == 0:
            print(f'Epoch {epoch+1} Batch {batch} Loss {train_loss.result():.4f} Acc {train_accuracy.result():.4f}')

    # Save checkpoint
    ckpt_save_path = ckpt_manager.save()
    print(f'Epoch {epoch+1} Loss {train_loss.result():.4f} Acc {train_accuracy.result():.4f}')
    print(f'Saved checkpoint: {ckpt_save_path}')
    print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs')

# -------------------------
# Greedy translation (inference)
# -------------------------
def evaluate(sentence):
    # sentence: string in source language (Portuguese)
    # returns token ids of translated sentence

    sentence = train_subwords_src.encode(sentence)
    sentence = [train_subwords_src.vocab_size] + sentence + [train_subwords_src.vocab_size + 1]
    encoder_input = tf.expand_dims(sentence, 0)

    output = tf.expand_dims([train_subwords_tgt.vocab_size], 0)

    for i in range(MAX_LENGTH):
        enc_padding_mask = create_padding_mask(encoder_input)
        look_ahead_mask = create_look_ahead_mask(tf.shape(output)[1])
        dec_padding_mask = create_padding_mask(encoder_input)

        predictions, attention_weights = transformer(encoder_input, output, training=False, enc_padding_mask=enc_padding_mask, look_ahead_mask=look_ahead_mask, dec_padding_mask=dec_padding_mask)

        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab)
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        # convert to python int for control flow
        pred_id_val = int(predicted_id[0, 0].numpy())
        if pred_id_val == (train_subwords_tgt.vocab_size + 1):
            break
        output = tf.concat([output, [[pred_id_val]]], axis=-1)

    return tf.squeeze(output, axis=0).numpy()

def translate(sentence):
    result_tokens = evaluate(sentence)
    # skip start token
    result_tokens = result_tokens[1:]
    # decode to string (tokens until end token)
    out = []
    for tok in result_tokens:
        if tok == train_subwords_tgt.vocab_size + 1:
            break
        out.append(tok)
    translated = train_subwords_tgt.decode(out)
    return translated

# -------------------------
# Test translation on some validation examples
# -------------------------
print("Testing translation on a few validation examples...")
for pt, en in val_examples.take(5):
    pt_str = pt.numpy().decode('utf-8')
    print('Source:', pt_str)
    print('Target:', en.numpy().decode('utf-8'))
    print('Predicted:', translate(pt_str))
    print('-----')

print('Done. You can increase NUM_LAYERS / D_MODEL and EPOCHS for better results.')
