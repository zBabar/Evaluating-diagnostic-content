import tensorflow as tf
import matplotlib.pyplot as plt
#import skimage.transform
import numpy as np
import time
import os
import cPickle as pickle
from scipy import ndimage
from core.utils import *
from core.bleu import evaluate
import random
import time

class CaptioningSolver(object):
    def __init__(self, model, data, val_data,test_data, **kwargs):
        """
        Required Arguments:
            - model: Show Attend and Tell caption generating model
            - data: Training data; dictionary with the following keys:
                - features: Feature vectors of shape (82783, 196, 512)
                - file_names: Image file names of shape (82783, )
                - captions: Captions of shape (400000, 17)
                - image_idxs: Indices for mapping caption to image of shape (400000, )
                - word_to_idx: Mapping dictionary from word to index
            - val_data: validation data; for print out BLEU scores for each epoch.
        Optional Arguments:
            - n_epochs: The number of epochs to run for training.
            - batch_size: Mini batch size.
            - update_rule: A string giving the name of an update rule
            - learning_rate: Learning rate; default value is 0.01.
            - print_every: Integer; training losses will be printed every print_every iterations.
            - save_every: Integer; model variables will be saved every save_every epoch.
            - pretrained_model: String; pretrained model path
            - model_path: String; model path for saving
            - test_model: String; model path for test
        """

        self.model = model
        self.data = data
        self.val_data = val_data
        self.test_data=test_data
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.update_rule = kwargs.pop('update_rule', 'adam')
        self.learning_rate = kwargs.pop('learning_rate', 0.01)
        self.print_bleu = kwargs.pop('print_bleu', False)
        self.print_every = kwargs.pop('print_every', 100)
        self.save_every = kwargs.pop('save_every', 1)
        self.log_path = kwargs.pop('log_path', './log/')
        self.model_path = kwargs.pop('model_path', './model/')
        self.pretrained_model = kwargs.pop('pretrained_model', None)
        self.test_model = kwargs.pop('test_model', './model/lstm/model-1')
        self.path = kwargs.pop('path', '/media/zaheer/Data/Image_Text_Datasets/IU_Xray/latest/Two_Images/Sample1')
        # set an optimizer by update rule
        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif self.update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)


    def train(self):
        # train/val dataset
        # Changed this because I keep less features than captions, see prepro
        #path = '/media/zaheer/Data/Image_Text_Datasets/IU_Xray/latest/Two_Images/word/Sample1/'
        n_examples = self.data['captions'].shape[0]
        #n_examples = self.data['features'].shape[0]
        n_iters_per_epoch = int(np.ceil(float(n_examples)/self.batch_size))
        features = self.data['features']

        captions = self.data['captions']
        image_idxs = self.data['image_idxs']
        val_features = self.val_data['features']
        val_captions = self.val_data['captions']
        n_iters_val = int(np.ceil(float(val_features.shape[0])/self.batch_size))

        # build graphs for training model and sampling captions
        # This scope fixed things!!
        with tf.variable_scope(tf.get_variable_scope()):
            loss = self.model.build_model()
            tf.get_variable_scope().reuse_variables()
            alph, bts, generated_captions= self.model.build_sampler(split='val',max_len=101)
            val_loss=self.model.valid_loss()

        # train op
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            early_optimizer = self.optimizer(learning_rate=self.learning_rate)
            mid_optimizer = self.optimizer(learning_rate=self.learning_rate / 10)
            late_optimizer = self.optimizer(learning_rate=self.learning_rate / 100)
            grads = tf.gradients(loss, tf.trainable_variables())
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_op = early_optimizer.apply_gradients(grads_and_vars=grads_and_vars)
            train_op0 = mid_optimizer.apply_gradients(grads_and_vars=grads_and_vars)
            train_op1 = late_optimizer.apply_gradients(grads_and_vars=grads_and_vars)

        # summary op
        val_loss_sum=tf.summary.scalar('val_batch_loss',val_loss) #### Added by Zaheer
        train_loss_sum = tf.summary.scalar('batch_loss', loss)
        # tf.scalar_summary('batch_loss', loss)
        # tf.summary.scalar('batch_loss', loss)
        # for var in tf.trainable_variables():
        #     #tf.histogram_summary(var.op.name, var)
        #     tf.summary.histogram(var.op.name, var)
        # for grad, var in grads_and_vars:
        #     #tf.histogram_summary(var.op.name+'/gradient', grad)
        #     tf.summary.histogram(var.op.name+'/gradient', grad)
        #
        # #summary_op = tf.merge_all_summaries()
        # summary_op = tf.summary.merge_all()

        print ("The number of epoch: %d" %self.n_epochs)
        print ("Data size: %d" %n_examples)
        print ("Batch size: %d" %self.batch_size)
        print ("Iterations per epoch: %d" %n_iters_per_epoch)

        #config = tf.ConfigProto(allow_soft_placement = True,device_count = {'GPU': 0}, log_device_placement=True)
        config = tf.ConfigProto(log_device_placement=True)
        run_options = tf.RunOptions()
        #config.gpu_options.per_process_gpu_memory_fraction=0.9
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            #summary_writer = tf.train.SummaryWriter(self.log_path, graph=tf.get_default_graph())
            summary_writer = tf.summary.FileWriter(self.log_path+'/train/', graph=tf.get_default_graph())
            val_summary_writer = tf.summary.FileWriter(self.log_path+'/val/', graph=tf.get_default_graph()) ### Added by Zaheer
            saver = tf.train.Saver(max_to_keep=100)# given value was 40

            if self.pretrained_model is not None:
                print ("Start training with pretrained Model..")
                saver.restore(sess, self.pretrained_model)

            prev_loss = -1
            curr_loss = 0
            start_t = time.time()

            for e in range(self.n_epochs):
                tik_epoch=time.time()
                print("epoch:",e)
                rand_idxs = np.random.permutation(n_examples)
                captions = captions[rand_idxs]
                image_idxs = image_idxs[rand_idxs]

                for i in range(n_iters_per_epoch):
                    tik_batch=time.time()
                    captions_batch = captions[i*self.batch_size:(i+1)*self.batch_size]
                    image_idxs_batch = image_idxs[i*self.batch_size:(i+1)*self.batch_size]
                    features_batch = features[image_idxs_batch]
                    feed_dict = {self.model.features: features_batch, self.model.captions: captions_batch}
                    print(i)
                    if (e <= 30):
                        _, l = sess.run([train_op, loss], feed_dict, options=run_options)
                    # elif (e<=60 and e>30):
                    #    _, l = sess.run([train_op0, loss], feed_dict)
                    else:
                        _, l = sess.run([train_op0, loss], feed_dict, options=run_options)
                    curr_loss += l
                    #print('Time to execute a batch:',time.time()-tik_batch)
                    # write summary for tensorboard visualization
                    if i % 10 == 0:
                        summary = sess.run(train_loss_sum, feed_dict)
                        summary_writer.add_summary(summary, e*n_iters_per_epoch + i)

                    if ((i+1) % self.print_every == 0):
                        print ("\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f" %(e+1, i+1, l))
                        ground_truths = captions[image_idxs == image_idxs_batch[0]]
                        decoded = decode_captions(ground_truths, self.model.idx_to_word)
                        for j, gt in enumerate(decoded):
                            print ("Ground truth %d: %s" %(j+1, gt))
                        gen_caps = sess.run(generated_captions, feed_dict)
                        decoded = decode_captions(gen_caps, self.model.idx_to_word)
                        print ("Generated caption: %s\n" %decoded[0])
                #print('Time to execute an epoch:', time.time() - tik_epoch)
                print ("Previous epoch loss: ", prev_loss)
                print ("Current epoch loss: ", curr_loss)
                print ("Elapsed time: ", time.time() - start_t)
                prev_loss = curr_loss
                curr_loss = 0
                curr_val_loss=0.0
                # print out BLEU scores and file write
                if self.print_bleu:
                    all_gen_cap = np.ndarray((val_features.shape[0], 101))
                    for i in range(n_iters_val):
                        val_features_batch = val_features[i*self.batch_size:(i+1)*self.batch_size]
                        val_captions_batch = val_captions[i * self.batch_size:(i + 1) * self.batch_size]
                        feed_dict = {self.model.features: val_features_batch,self.model.captions:val_captions_batch}
                        gen_cap,val_l = sess.run([generated_captions,val_loss], feed_dict=feed_dict)#### Added by Zaheer
                        all_gen_cap[i*self.batch_size:(i+1)*self.batch_size] = gen_cap
                        curr_val_loss+=val_l ### Added by Zaheer
                        if i % 10 == 0:
                            val_summary = sess.run(val_loss_sum, feed_dict)
                            val_summary_writer.add_summary(val_summary, e * n_iters_val + i)
                    print(curr_val_loss)
                    #### End
                    all_decoded = decode_captions(all_gen_cap, self.model.idx_to_word)
                    GroundTruth_decoded = decode_captions(self.val_data['captions'], self.model.idx_to_word)
                    print("Validation Sample Generated Caption")
                    sample_gen=random.randint(0,val_captions.shape[0]-1)
                    print("Ground truth: %s"%(GroundTruth_decoded[sample_gen]))
                    print("Generated caption: %s\n" %all_decoded[sample_gen])
                    save_pickle(all_decoded, self.path+"test/test.candidate.captions.pkl")
                    scores = evaluate(data_path=self.path, split='test', get_scores=True)
                    write_bleu(scores=scores, path=self.path, epoch=e)

                # save model's parameters
                if (e+1) % self.save_every == 0:
                    saver.save(sess, os.path.join(self.model_path, 'model'), global_step=e+1)
                    print ("model-%s saved." %(e+1))

            #self.test(self.test_data,alphas=alph, betas=bts, sampled_captions=generated_captions)
        tf.reset_default_graph()


    # def test(self,data,alphas,betas,sampled_captions, split='test', attention_visualization=False, save_sampled_captions=True):
    #     '''
    #     Args:
    #         - data: dictionary with the following keys:
    #         - features: Feature vectors of shape (5000, 196, 512)
    #         - file_names: Image file names of shape (5000, )
    #         - captions: Captions of shape (24210, 17)
    #         - image_idxs: Indices for mapping caption to image of shape (24210, )
    #         - features_to_captions: Mapping feature to captions (5000, 4~5)
    #         - split: 'train', 'val' or 'test'
    #         - attention_visualization: If True, visualize attention weights with images for each sampled word. (ipthon notebook)
    #         - save_sampled_captions: If True, save sampled captions to pkl file for computing BLEU scores.
    #     '''
    #
    #     features = data['features']
    #
    #     # build a graph to sample captions
    #     #alphas, betas, sampled_captions = self.model.build_sampler(split='test',max_len=101)    # (N, max_len, L), (N, max_len)
    #
    #     config = tf.ConfigProto(allow_soft_placement=True)
    #     config.gpu_options.allow_growth = True
    #     with tf.Session(config=config) as sess:
    #         saver = tf.train.Saver()
    #         saver.restore(sess, self.test_model)
    #         features_batch, image_files = sample_coco_minibatch(data, self.batch_size)
    #         feed_dict = { self.model.features: features_batch }
    #         alps, bts, sam_cap = sess.run([alphas, betas, sampled_captions], feed_dict)  # (N, max_len, L), (N, max_len)
    #         decoded = decode_captions(sam_cap, self.model.idx_to_word)
    #
    #         if attention_visualization:
    #             for n in range(10):
    #                 print "Sampled Caption: %s" %decoded[n]
    #
    #                 # Plot original image
    #                 img = ndimage.imread(image_files[n])
    #                 plt.subplot(4, 5, 1)
    #                 plt.imshow(img)
    #                 plt.axis('off')
    #
    #                 # Plot images with attention weights
    #                 words = decoded[n].split(" ")
    #                 for t in range(len(words)):
    #                     if t > 18:
    #                         break
    #                     plt.subplot(4, 5, t+2)
    #                     plt.text(0, 1, '%s(%.2f)'%(words[t], bts[n,t]) , color='black', backgroundcolor='white', fontsize=8)
    #                     plt.imshow(img)
    #                     alp_curr = alps[n,t,:].reshape(14,14)
    #                     alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=16, sigma=20)
    #                     plt.imshow(alp_img, alpha=0.85)
    #                     plt.axis('off')
    #                 plt.show()
    #
    #         if save_sampled_captions:
    #             all_sam_cap = np.ndarray((features.shape[0], 101))
    #             num_iter = int(np.ceil(float(features.shape[0]) / self.batch_size))
    #             for i in range(num_iter):
    #                 features_batch = features[i*self.batch_size:(i+1)*self.batch_size]
    #                 feed_dict = { self.model.features: features_batch }
    #                 all_sam_cap[i*self.batch_size:(i+1)*self.batch_size] = sess.run(sampled_captions, feed_dict)
    #             all_decoded = decode_captions(all_sam_cap, self.model.idx_to_word)
    #             save_pickle(all_decoded, "./data/%s/%s.candidate.captions.pkl" %(split,split))
    #             scores = evaluate(data_path='./data', split='test', get_scores=True)
    #             write_bleu(scores=scores, path=self.model_path, epoch=0)
