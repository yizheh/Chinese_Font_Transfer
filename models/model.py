# %%
import tensorflow as tf
import numpy as np
import matplotlib as plt
from PIL import Image
from data_preprocess import data_preprocess, Dataset
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# %%
# hyperparameters
image_dim = (64, 64, 1)
learning_rate = 0.005
batch_size = 32
element_num = batch_size*image_dim[0]*image_dim[1]*image_dim[2]
epochs = 30
beta1 = 0.9
lambda_p = 1
lambda_1 = 10
lambda_2 = 1
lambda_3 = 1
lambda_4 = 30
print_every = 1


# %%

def leaky_relu(x):
    alpha = -0.1
    return tf.where( x >= 0.0, x, alpha*x )

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale*tf.where(x >= 0.0, x, alpha*tf.nn.elu(x))


# %%
def generator(source_font, reuse=False, training=True):

    with tf.variable_scope('generator', reuse=reuse):
        # input tensor size is 64*64*1
        input_layer = tf.reshape(source_font, (-1, 64, 64, 1))

        # 64*64*1 -> 64*64*64
        conv1 = tf.layers.conv2d(
            inputs=input_layer, filters=64, kernel_size=3, padding='same')
        conv1 = tf.layers.batch_normalization(conv1, training=True)
        conv1 = leaky_relu(conv1)

        # 128*128*64 -> 64*64*128
        conv2 = tf.layers.conv2d(
            inputs=conv1, filters=64, kernel_size=3, strides=2, padding='same')
        conv2 = tf.layers.batch_normalization(conv2, training=True)
        conv2 = leaky_relu(conv2)
        conv2 = tf.layers.conv2d(
            inputs=conv2, filters=128, kernel_size=3, padding='same')
        conv2 = tf.layers.batch_normalization(conv2, training=True)
        conv2 = leaky_relu(conv2)

        # 64*64*128 -> 32*32*256
        conv3 = tf.layers.conv2d(
            inputs=conv2, filters=128, kernel_size=3, strides=2, padding='same')
        conv3 = tf.layers.batch_normalization(conv3, training=True)
        conv3 = leaky_relu(conv3)
        conv3 = tf.layers.conv2d(
            inputs=conv3, filters=256, kernel_size=3, padding='same')
        conv3 = tf.layers.batch_normalization(conv3, training=True)
        conv3 = leaky_relu(conv3)

        # 32*32*256 -> 16*16*512
        conv4 = tf.layers.conv2d(
            inputs=conv3, filters=256, kernel_size=3, strides=2, padding='same')
        conv4 = tf.layers.batch_normalization(conv4, training=True)
        conv4 = leaky_relu(conv4)
        conv4 = tf.layers.conv2d(
            inputs=conv4, filters=512, kernel_size=3, padding='same')
        conv4 = tf.layers.batch_normalization(conv4, training=True)
        conv4 = leaky_relu(conv4)

        # 16*16*512 -> 8*8*512
        conv5 = tf.layers.conv2d(
            inputs=conv4, filters=512, kernel_size=3, strides=2, padding='same')
        conv5 = tf.layers.batch_normalization(conv5, training=True)
        conv5 = leaky_relu(conv5)

        # 8*8*512 -> 16*16*512
        # 16*16*512 + 16*16*512 -> 16*16*1024
        # 16*16*1024 -> 16*16*512
        # 16*16*512 -> 16*16*256
        up6 = tf.layers.conv2d_transpose(
            inputs=conv5, filters=512, kernel_size=3, strides=2, padding='same')
        up6 = tf.layers.batch_normalization(up6, training=True)
        up6 = leaky_relu(up6)
        up6 = tf.concat([conv4, up6], axis=3)
        conv6 = tf.layers.conv2d(
            inputs=up6, filters=512, kernel_size=3, padding='same')
        conv6 = tf.layers.batch_normalization(conv6, training=True)
        conv6 = leaky_relu(conv6)
        conv6 = tf.layers.conv2d(
            inputs=conv6, filters=256, kernel_size=3, padding='same')
        conv6 = tf.layers.batch_normalization(conv6, training=True)
        conv6 = leaky_relu(conv6)

        # 16*16*256 -> 32*32*256
        # 32*32*256 + 32*32*256 -> 32*32*512
        # 32*32*512 -> 32*32*256
        # 32*32*256 -> 32*32*128
        up7 = tf.layers.conv2d_transpose(
            inputs=conv6, filters=256, kernel_size=3, strides=2, padding='same')
        up7 = tf.layers.batch_normalization(up7, training=True)
        up7 = leaky_relu(up7)
        up7 = tf.concat([conv3, up7], axis=3)
        conv7 = tf.layers.conv2d(
            inputs=up7, filters=256, kernel_size=3, padding='same')
        conv7 = tf.layers.batch_normalization(conv7, training=True)
        conv7 = leaky_relu(conv7)
        conv7 = tf.layers.conv2d(
            inputs=conv7, filters=128, kernel_size=3, padding='same')
        conv7 = tf.layers.batch_normalization(conv7, training=True)
        conv7 = leaky_relu(conv7)

        # 32*32*128 -> 64*64*128
        # 64*64*128 + 64*64*128 -> 64*64*256
        # 64*64*256 -> 64*64*128 -> 64*64*64
        up8 = tf.layers.conv2d_transpose(
            inputs=conv7, filters=128, kernel_size=3, strides=2, padding='same')
        up8 = tf.layers.batch_normalization(up8, training=True)
        up8 = leaky_relu(up8)
        up8 = tf.concat([conv2, up8], axis=3)
        conv8 = tf.layers.conv2d(
            inputs=up8, filters=128, kernel_size=3, padding='same')
        conv8 = tf.layers.batch_normalization(conv8, training=True)
        conv8 = leaky_relu(conv8)
        conv8 = tf.layers.conv2d(
            inputs=conv8, filters=64, kernel_size=3, padding='same')
        conv8 = tf.layers.batch_normalization(conv8, training=True)
        conv8 = leaky_relu(conv8)

        # 64*64*64 -> 128*128*64
        # 128*128*64 + 128*128*64 -> 128*128*128
        # 128*128*128 -> 128*128*64 -> 128*128*1
        up9 = tf.layers.conv2d_transpose(
            inputs=conv8, filters=64, kernel_size=3, strides=2, padding='same')
        up9 = tf.layers.batch_normalization(up9, training=True)
        up9 = leaky_relu(up9)
        up9 = tf.concat([conv1, up9], axis=3)
        conv9 = tf.layers.conv2d(
            inputs=up9, filters=64, kernel_size=3, padding='same')
        conv9 = tf.layers.batch_normalization(conv9, training=True)
        conv9 = leaky_relu(conv9)
        conv9 = tf.layers.conv2d(
            inputs=conv9, filters=1, kernel_size=3, padding='same')
        conv9 = tf.layers.batch_normalization(conv9, training=True)
        conv9 = tf.nn.sigmoid(conv9)

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
        conv1 = leaky_relu(conv1)

        conv2 = tf.layers.conv2d(
            inputs=conv1, filters=512, kernel_size=3, strides=2, padding='same')
        conv2 = tf.layers.batch_normalization(conv2, training=True)
        conv2 = leaky_relu(conv2)

        conv3 = tf.layers.conv2d(
            inputs=conv2, filters=1024, kernel_size=3, strides=2, padding='same')
        conv3 = tf.layers.batch_normalization(conv3, training=True)
        conv3 = leaky_relu(conv3)

        conv4 = tf.layers.conv2d(
            inputs=conv3, filters=2048, kernel_size=3, strides=2, padding='same')
        conv4 = tf.layers.batch_normalization(conv4, training=True)
        conv4 = leaky_relu(conv4)

        flat = tf.reshape(conv4, (-1, 8*8*2048))
        logits = tf.layers.dense(flat, 1)
        out = tf.nn.sigmoid(logits)

        return out


# %%
class TransNN:
    def __init__(self):
        tf.reset_default_graph()

    '''def input_setup(self, image_dim):
        
        self.target_fonts = tf.placeholder(
            tf.float32, (None,  128, 128, 1), name='target_font')
        self.source_fonts = tf.placeholder(
            tf.float32, (None,  128, 128, 1), name='source_font')'''

    def model_setup(self):
        '''
        This function sets up the model to train
        '''
        self.target_fonts = tf.placeholder(
            tf.float32, (None,  64, 64, 1), name='target_fonts')
        self.source_fonts = tf.placeholder(
            tf.float32, (None,  64, 64, 1), name='source_fonts')

        self.generated_fonts = generator(self.source_fonts)
        self.real_score = discriminator(self.target_fonts)
        self.fake_score = discriminator(self.generated_fonts, reuse=True)

    def model_loss(self):
        #pix2pix cross_entropy loss
        p2p_cross_entropy = tf.reduce_sum(- lambda_p * np.multiply(self.target_fonts, tf.log(self.generated_fonts)
                                                         ) - np.multiply(1-self.target_fonts, tf.log(1-self.generated_fonts)))
        diffrence = tf.subtract(self.target_fonts, self.generated_fonts)
        #l1-norm
        l1_norm = tf.reduce_sum( tf.abs(diffrence))
        #l2-norm
        l2_norm = tf.reduce_sum( np.multiply( diffrence, diffrence))

        v1_loss = lambda_1 * p2p_cross_entropy + lambda_4 * l1_norm 
        v1_loss /= element_num
        #v1_loss = v1_loss / target_font.shape[0]

        # v2_loss evaluates the discriminator performence of seperating generated fonts from target fonts
        v2_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_score, labels=tf.ones_like(
            self.real_score))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_score, labels=tf.zeros_like(self.fake_score)))

        # v3_loss evaluate the generator performence of generating real fonts
        v3_loss_log = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.fake_score, labels=tf.ones_like(self.fake_score)))
        # in case of gradient varnishment, we use v3_loss without log
        v3_loss = tf.reduce_sum(self.fake_score)


        #self.generator_loss =  ( lambda_1 * v1_loss + lambda_2 * v3_loss_log ) / batch_size
        #self.generator_loss =  ( lambda_1 * v1_loss + lambda_2 *( batch_size -  v3_loss ) ) / batch_size
        self.generator_loss =  v1_loss  / batch_size 

        self.discriminator_loss = ( lambda_3 * v2_loss ) / batch_size

        self.model_vars = tf.trainable_variables()

        d_vars = [var for var in self.model_vars if var.name.startswith(
            'discriminator')]
        g_vars = [
            var for var in self.model_vars if var.name.startswith('generator')]

        self.d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(
            self.discriminator_loss, var_list=d_vars)
        self.g_train_opt = tf.train.AdamOptimizer(
            learning_rate, beta1=beta1).minimize(self.generator_loss, var_list=g_vars)

    def train(self):
        self.model_setup()

        self.model_loss()

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init)

            # Training
            for epoch_no in range(epochs):
                dataset.shuffle_data()

                batch_no = 0
                for source_font, target_font in dataset.get_batches(batch_size):
                    batch_no += 1
                    print("Batch No: {}/{} ".format(batch_no,
                                                    dataset.train_num // batch_size))

                    # Use generator to generate fonts from source fonts
                    sess.run(self.generated_fonts, feed_dict={
                             self.source_fonts: source_font, self.target_fonts: target_font})

                    # Optimize the discriminator
                    sess.run(self.d_train_opt, feed_dict={self.target_fonts: target_font, self.source_fonts: source_font})

                    # Optimize the generator
                    sess.run(self.g_train_opt, feed_dict={
                             self.target_fonts: target_font, self.source_fonts: source_font})

                    if epoch_no % print_every == 0:
                        train_loss_d = self.discriminator_loss.eval(
                            {self.target_fonts: target_font, self.source_fonts: source_font})
                        train_loss_g = self.generator_loss.eval(
                            {self.source_fonts: source_font, self.target_fonts: target_font})

                        print("Epoch {}/{}...".format(epoch_no+1, epochs),
                              "Discriminator Loss: {:.8f}...".format(
                                  train_loss_d),
                              "Generator Loss: {:.8f}".format(train_loss_g))
                        # Save losses to view after training
                        #losses.append((train_loss_d, train_loss_g))

            saver.save(sess, './checkpoints/generator')

    def test(self):
        print("Testing the results")

        # self.model_setup()

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init)

            saver = tf.train.import_meta_graph('./checkpoints/generator.meta')
            saver.restore(sess, tf.train.latest_checkpoint('./checkpoints/'))
            graph = tf.get_default_graph()
            image_no = 0

            for num in range(0, dataset.test_num, batch_size):
                generated_fonts = sess.run(self.generated_fonts, feed_dict={
                                           self.source_fonts: dataset.test['source_font'][num:num+batch_size], self.target_fonts: dataset.test['target_font'][num:num+batch_size]})
                source_fonts = dataset.test['target_font'][num:num+batch_size]

                for batch_no in range(min(batch_size, generated_fonts.shape[0])):
                    generated_font = generated_fonts[batch_no]
                    target_font = source_fonts[batch_no]

                    mid = np.append(generated_font, generated_font, axis=2)
                    mid = np.append(generated_font, mid, axis=2)
                    mid = mid*255
                    mid = mid.astype('uint8')
                    img = Image.fromarray(mid)
                    img.save(fp='./generated_fonts/generated_' +
                             str(image_no) + '.jpg')

                    mid[:, :, :] = target_font
                    mid = mid*255
                    mid = mid.astype('uint8')
                    img = Image.fromarray(mid)
                    img.save(fp='./generated_fonts/target_' +
                             str(image_no)+'.jpg')

                    image_no += 1


# %%
# writer=tf.summary.FileWriter(r"/Users/hangyizhe/GitHub/Chinese_Font_Transfer/models",tf.get_default_graph())
# writer.close()
source_fonts, target_fonts = data_preprocess()
dataset = Dataset(source_fonts, target_fonts, test_frac=0.1, val_frac=0)

# %%
net = TransNN()
net.train()
print("Training Finished!")

net.test()

print("Test Finished!")


# %%
