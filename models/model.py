# %%
import tensorflow as tf
import numpy as np
import matplotlib as plt
from PIL import Image
from data_preprocess import data_preprocess, Dataset
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# %%
# hyperparameters
image_dim = (128, 128, 1)
learning_rate = 0.00001
batch_size = 32
element_num = batch_size*image_dim[0]*image_dim[1]*image_dim[2]
epochs = 5
beta1 = 0.9
lambda_p = 0.1
lambda_1 = 0.1
lambda_2 = 0.1
lambda_3 = 0.1

#%%



#%%
def model_inputs(image_dim):

    target_fonts = tf.placeholder(
        tf.float32, (None,  128, 128, 1), name='target_font')
    source_fonts = tf.placeholder(
        tf.float32, (None,  128, 128, 1), name='source_font')

    return target_fonts, source_fonts

# %%


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale*tf.where(x >= 0.0, x, alpha*tf.nn.elu(x))


# %%
def generator(source_font, reuse=False, training=True):
    with tf.variable_scope('generator', reuse=reuse):
        w_init = tf.contrib.layers.xavier_initializer()

        # input tensor size is 128*128*1
        input_layer = tf.reshape(source_font, (-1, 128, 128, 1))

        # 128*128*1 -> 128*128*64
        conv1 = tf.layers.conv2d(
            inputs=input_layer, filters=64, kernel_size=3, padding='same')
        conv1 = tf.layers.batch_normalization(conv1, training=True)
        conv1 = selu(conv1)

        # 128*128*64 -> 64*64*128
        conv2 = tf.layers.conv2d(
            inputs=conv1, filters=64, kernel_size=3, strides=2, padding='same')
        conv2 = tf.layers.batch_normalization(conv2, training=True)
        conv2 = selu(conv2)
        conv2 = tf.layers.conv2d(
            inputs=conv2, filters=128, kernel_size=3, padding='same')
        conv2 = tf.layers.batch_normalization(conv2, training=True)
        conv2 = selu(conv2)

        # 64*64*128 -> 32*32*256
        conv3 = tf.layers.conv2d(
            inputs=conv2, filters=128, kernel_size=3, strides=2, padding='same')
        conv3 = tf.layers.batch_normalization(conv3, training=True)
        conv3 = selu(conv3)
        conv3 = tf.layers.conv2d(
            inputs=conv3, filters=256, kernel_size=3, padding='same')
        conv3 = tf.layers.batch_normalization(conv3, training=True)
        conv3 = selu(conv3)

        # 32*32*256 -> 16*16*512
        conv4 = tf.layers.conv2d(
            inputs=conv3, filters=256, kernel_size=3, strides=2, padding='same')
        conv4 = tf.layers.batch_normalization(conv4, training=True)
        conv4 = selu(conv4)
        conv4 = tf.layers.conv2d(
            inputs=conv4, filters=512, kernel_size=3, padding='same')
        conv4 = tf.layers.batch_normalization(conv4, training=True)
        conv4 = selu(conv4)

        # 16*16*512 -> 8*8*512
        conv5 = tf.layers.conv2d(
            inputs=conv4, filters=512, kernel_size=3, strides=2, padding='same')
        conv5 = tf.layers.batch_normalization(conv5, training=True)
        conv5 = selu(conv5)

        # 8*8*512 -> 16*16*512
        # 16*16*512 + 16*16*512 -> 16*16*1024
        # 16*16*1024 -> 16*16*512
        # 16*16*512 -> 16*16*256
        up6 = tf.layers.conv2d_transpose(
            inputs=conv5, filters=512, kernel_size=3, strides=2, padding='same')
        up6 = tf.layers.batch_normalization(up6, training=True)
        up6 = selu(up6)
        up6 = tf.concat([conv4, up6], axis=3)
        conv6 = tf.layers.conv2d(
            inputs=up6, filters=512, kernel_size=3, padding='same')
        conv6 = tf.layers.batch_normalization(conv6, training=True)
        conv6 = selu(conv6)
        conv6 = tf.layers.conv2d(
            inputs=conv6, filters=256, kernel_size=3, padding='same')
        conv6 = tf.layers.batch_normalization(conv6, training=True)
        conv6 = selu(conv6)

        # 16*16*256 -> 32*32*256
        # 32*32*256 + 32*32*256 -> 32*32*512
        # 32*32*512 -> 32*32*256
        # 32*32*256 -> 32*32*128
        up7 = tf.layers.conv2d_transpose(
            inputs=conv6, filters=256, kernel_size=3, strides=2, padding='same')
        up7 = tf.layers.batch_normalization(up7, training=True)
        up7 = selu(up7)
        up7 = tf.concat([conv3, up7], axis=3)
        conv7 = tf.layers.conv2d(
            inputs=up7, filters=256, kernel_size=3, padding='same')
        conv7 = tf.layers.batch_normalization(conv7, training=True)
        conv7 = selu(conv7)
        conv7 = tf.layers.conv2d(
            inputs=conv7, filters=128, kernel_size=3, padding='same')
        conv7 = tf.layers.batch_normalization(conv7, training=True)
        conv7 = selu(conv7)

        # 32*32*128 -> 64*64*128
        # 64*64*128 + 64*64*128 -> 64*64*256
        # 64*64*256 -> 64*64*128 -> 64*64*64
        up8 = tf.layers.conv2d_transpose(
            inputs=conv7, filters=128, kernel_size=3, strides=2, padding='same')
        up8 = tf.layers.batch_normalization(up8, training=True)
        up8 = selu(up8)
        up8 = tf.concat([conv2, up8], axis=3)
        conv8 = tf.layers.conv2d(
            inputs=up8, filters=128, kernel_size=3, padding='same')
        conv8 = tf.layers.batch_normalization(conv8, training=True)
        conv8 = selu(conv8)
        conv8 = tf.layers.conv2d(
            inputs=conv8, filters=64, kernel_size=3, padding='same')
        conv8 = tf.layers.batch_normalization(conv8, training=True)
        conv8 = selu(conv8)

        # 64*64*64 -> 128*128*64
        # 128*128*64 + 128*128*64 -> 128*128*128
        # 128*128*128 -> 128*128*64 -> 128*128*1
        up9 = tf.layers.conv2d_transpose(
            inputs=conv8, filters=64, kernel_size=3, strides=2, padding='same')
        up9 = tf.layers.batch_normalization(up9, training=True)
        up9 = selu(up9)
        up9 = tf.concat([conv1, up9], axis=3)
        conv9 = tf.layers.conv2d(
            inputs=up9, filters=64, kernel_size=3, padding='same')
        conv9 = tf.layers.batch_normalization(conv9, training=True)
        conv9 = selu(conv9)
        conv9 = tf.layers.conv2d(
            inputs=conv9, filters=1, kernel_size=3, padding='same')
        conv9 = tf.layers.batch_normalization(conv9, training=True)
        conv9 = tf.nn.sigmoid(conv9,)

        return conv9


# %%
def discriminator(x, reuse=False):
    """
        x: a real image of target fonts or a fake image generated by generator use source fonts
    """
    with tf.variable_scope('discriminator', reuse=reuse):
        w_init = tf.contrib.layers.xavier_initializer()

        conv1 = tf.layers.conv2d(
            inputs=x, filters=256, kernel_size=3, strides=2, padding='same')
        conv1 = tf.layers.batch_normalization(conv1, training=True)
        conv1 = selu(conv1)

        conv2 = tf.layers.conv2d(
            inputs=conv1, filters=512, kernel_size=3, strides=2, padding='same')
        conv2 = tf.layers.batch_normalization(conv2, training=True)
        conv2 = selu(conv2)

        conv3 = tf.layers.conv2d(
            inputs=conv2, filters=1024, kernel_size=3, strides=2, padding='same')
        conv3 = tf.layers.batch_normalization(conv3, training=True)
        conv3 = selu(conv3)

        conv4 = tf.layers.conv2d(
            inputs=conv3, filters=2048, kernel_size=3, strides=2, padding='same')
        conv4 = tf.layers.batch_normalization(conv4, training=True)
        conv4 = selu(conv4)

        flat = tf.reshape(conv4, (-1, 8*8*2048))
        logits = tf.layers.dense(flat, 1)
        out = tf.nn.sigmoid(logits)

        return out


# %%
def model_loss(target_font, generated_font, element_num, lambda_p, lambda_1, lambda_2, lambda_3):
    real_score = discriminator(target_font)
    fake_score = discriminator(generated_font, reuse=True)

    # v1_loss is the pix2pix loss
    v1_loss = tf.reduce_sum(- lambda_p * np.multiply(target_font, tf.log(generated_font)
                                                     ) - np.multiply(1-target_font, tf.log(1-generated_font)))
    v1_loss /= element_num
    #v1_loss = v1_loss / target_font.shape[0]

    # v2_loss evaluates the discriminator performence of seperating generated fonts from target fonts
    v2_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_score, labels=tf.ones_like(
        real_score))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_score, labels=tf.zeros_like(fake_score)))

    # v3_loss evaluate the generator performence of generating real fonts
    v3_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=fake_score, labels=tf.ones_like(fake_score)))

    generator_loss = lambda_1 * v1_loss + lambda_2 * v3_loss
    discriminator_loss = lambda_3 * v2_loss

    return discriminator_loss, generator_loss


# %%

def model_opt(generator_loss, discriminator_loss, learning_rate, beta1):
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(
            discriminator_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(
            learning_rate, beta1=beta1).minimize(generator_loss, var_list=g_vars)

    return d_train_opt, g_train_opt


# %%
class GAN:
    def __init__(self, image_dim, element_num, learning_rate, lambda_p, lambda_1, lambda_2, lambda_3, beta1=0.5):
        tf.reset_default_graph()

        self.target_font, self.source_font = model_inputs(image_dim)

        self.generated_font = generator(self.source_font)

        self.d_loss, self.g_loss = model_loss(
            self.target_font, self.generated_font, element_num, lambda_p, lambda_1, lambda_2, lambda_3)

        self.d_opt, self.g_opt = model_opt(
            self.g_loss, self.d_loss, learning_rate, beta1)


# %%
def train(net, dataset, epochs, batch_size, print_every=5):
    #sample_z = data_generator()

    samples, losses = [], []
    steps = 0
    a = np.zeros((1, 128, 128, 1))
    a = a.astype(np.float32)

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        img_generated = tf.Variable(
            tf.random_normal(shape=[batch_size, 128, 128, 1]), name='font_generator')
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            for source_font, target_font in dataset.get_batches(batch_size):
                steps += 1
                print(steps)

                img_generated = net.generated_font
                sess.run(img_generated, feed_dict={
                    net.target_font: target_font, net.source_font: source_font})

                _ = sess.run(net.d_opt, feed_dict={
                             net.target_font: target_font, net.source_font: source_font})
                _ = sess.run(net.g_opt, feed_dict={
                             net.target_font: target_font, net.source_font: source_font})

                if steps % print_every == 0:
                    train_loss_d = net.d_loss.eval(
                        {net.target_font: target_font, net.source_font: source_font})
                    train_loss_g = net.g_loss.eval(
                        {net.source_font: source_font, net.target_font: target_font})

                    print("Epoch {}/ {}...".format(e+1, epochs),
                          "Discriminator Loss: {:.8f}...".format(train_loss_d),
                          "Generator Loss: {:.8f}".format(train_loss_g))
                    # Save losses to view after training
                    losses.append((train_loss_d, train_loss_g))

        saver.save(sess, './checkpoints/generator')

        return losses

# %%


def test(net, dataset, print_every=5):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver = tf.train.import_meta_graph('./checkpoints/generator.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./checkpoints/'))

        graph = tf.get_default_graph()
        source_font = graph.get_tensor_by_name('source_font:0')
        target_font = graph.get_tensor_by_name('target_font:0')
        font_generator = graph.get_tensor_by_name('font_generator:0')
        sess.run(tf.global_variables_initializer())

        generated_fonts = sess.run(font_generator, feed_dict={
            source_font: dataset.test['source_font'], target_font: dataset.test['target_font']})

        print(generated_fonts.shape, target_font.shape )

        num = 1
        mid = np.zeros((128, 128, 3))
        for num in range(dataset.test_num):
            generated_font = generated_fonts[num]
            target_font = dataset.test['target_font'][num]
            mid[:, :, :] = generated_font
            mid = mid*255
            mid = mid.astype('uint8')
            img = Image.fromarray(mid)
            img.save(fp='./generated_fonts/generated_' + str(num) + '.jpg')

            mid[:, :, :] = target_font
            mid = mid*255
            mid = mid.astype('uint8')
            img = Image.fromarray(mid)
            img.save(fp='./generated_fonts/target_'+str(num)+'.jpg')
            num += 1


# %%



# %%


# %%
# writer=tf.summary.FileWriter(r"/Users/hangyizhe/GitHub/Chinese_Font_Transfer/models",tf.get_default_graph())
# writer.close()
source_fonts, target_fonts = data_preprocess()
dataset = Dataset(source_fonts, target_fonts, test_frac=0.1, val_frac=0)

net = GAN((128, 128, 1), element_num, learning_rate,
          lambda_p, lambda_1, lambda_2, lambda_3)
losses = train(net, dataset, epochs, batch_size)
print("Training Finished!")

test(net, dataset)

# %%


# %%
