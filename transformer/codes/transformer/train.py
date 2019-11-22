from transformer import *
from utils import *
import tensorflow as tf
import numpy
import time
from jiwer import wer as jwer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class parameters():
    number_sentence = 3
    num_layers = 4
    d_model = 90
    dff = 1024
    num_heads = 15
    input_vocab_size = 0
    target_vocab_size = 0
    dropout_rate = 0.1
    freq = 100
    feature = 1
    data = 'eeg'
    seed = 1234
    n_batches = 10
    epochs = 50

def train(number_sentence):
    """
    Get dataset and parameters
    """
    params = parameters()
    tf.random.set_seed(params.seed)
    params.number_sentence = number_sentence
    input_set, target_set, seq_len_set, original_set = load_data(params)
    input_set, _ = pad_sequences(input_set, dtype=np.float32)
    target_set, _ = pad_sequences(target_set, dtype=np.int64)
    params.d_model = input_set.shape[-1]
    params.target_vocab_size = len(params.dictionary)+1
    params.max_length = len(target_set[0])
    x_train, x_test, y_train, y_test = train_test_split(input_set, target_set, test_size = 0.2, random_state = 42)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.2, random_state=42)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(
        8192, seed=params.seed).batch(1)
    val_dataset =  tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(
        8192, seed=params.seed).batch(1)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(
        8192, seed=params.seed).batch(params.n_batches)
    """
        Define loss, model, optimizer
    """
    learning_rate = CustomSchedule(params.d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='val_accuracy')
    transformer = Transformer(
        num_layers=params.num_layers,
        d_model=params.d_model, num_heads=params.num_heads,
        dff=params.dff, target_vocab_size=params.target_vocab_size,
        pe_target=6000)
    checkpoint_path = "./checkpoints/train"

    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    # if ckpt_manager.latest_checkpoint:
    #     ckpt.restore(ckpt_manager.latest_checkpoint)
    #     print('Latest checkpoint restored!!')
    # The @tf.function trace-compiles train_step into a TF graph for faster
    # execution. The function specializes to the precise shape of the argument
    # tensors. To avoid re-tracing due to the variable sequence lengths or variable
    # batch sizes (the last batch is smaller), use input_signature to specify
    # more generic shapes.



    @tf.function
    def train_step(inp,tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        # enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
        combined_mask = create_combined_mask(tar=tar_inp)
        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar_inp,
                                         True,
                                         None,
                                         combined_mask,
                                         None)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(tar_real, predictions)
    @tf.function
    def eval_step(inp,tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        # enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
        combined_mask = create_combined_mask(tar=tar_inp)
        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar_inp,
                                         True,
                                         None,
                                         combined_mask,
                                         None)
            loss = loss_function(tar_real, predictions)

        val_loss(loss)
        val_accuracy(tar_real, predictions)
    for epoch in range(params.epochs):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()
        # inp -> portuguese, tar -> english
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)
        for (batch, (inp, tar)) in enumerate(train_dataset):
            eval_step(inp, tar)
        if batch % 50 == 0:
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()))

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

        print('Epoch {} Train Loss {:.4f} Train Accuracy {:.4f} Val Loss {:.4f} Val Accuracy {:.4f}'.format(
                                                            epoch + 1,
                                                            train_loss.result(),
                                                            train_accuracy.result(),
                                                            val_loss.result(),
                                                            val_accuracy.result()))

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
    """
        Evaluation + Test
    """
    params.transformer = transformer
    ground_truth = []
    predicted = []
    wer = None
    for inp,tar in test_dataset:
        gtruth, pred = translate(inp,tar, params, True, False)
        ground_truth.append(gtruth)
        predicted.append(pred)
    wer = jwer(ground_truth,predicted)
    print("word error rate : {}".format(wer))
if __name__ == "__main__":
    number_sentence = 3
    train(number_sentence)