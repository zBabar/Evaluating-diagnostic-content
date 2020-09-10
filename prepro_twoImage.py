from scipy import ndimage
from collections import Counter, OrderedDict
from core.vggnet import Vgg19
from core.utils import *
from Main_Hyper import Main_hyper
import tensorflow as tf
import numpy as np
import pandas as pd
import hickle
import nltk


# Defining function to merg to image features image1+image2
def merge_feature_maps(comb_map):

    numOfMaps=comb_map.shape[0]

    i=0
    if numOfMaps ==1:
        return comb_map
    else:
       
        new_map = np.hstack((comb_map[i, :], comb_map[i + 1, :]))

    return new_map


def _process_caption_data(caption_data,max_length):
    caption_data=pd.read_json(caption_data)
    images_lst = caption_data['images'][0]

    # id_to_filename is a dictionary such as {image_id: filename]}
    # id_to_filename = {image['image_id']: image['image_id'] for image in caption_data}

    # data is a list of dictionary which contains 'captions', 'file_name' and 'image_id' as key.


    del_idx = []
    for i, caption in enumerate(caption_data['caption']):
        sents=nltk.sent_tokenize(caption)
        temp = []
        temp.append(sents[-1])
        sents=temp+sents[:-1]
        caption=' '.join(sents)#TD.detokenize(sents)
        caption = caption.replace(',', '').replace("'", "").replace('"', '').replace('.', '')
        caption = caption.replace('&', 'and').replace('(', '').replace(")", "").replace('-', ' ')
        caption = " ".join(caption.split())  # replace multiple spaces


        if len(nltk.word_tokenize(caption)) > max_length:
        #if len(nltk.sent_tokenize(caption)) > 6:
            # caption_data['caption'][i]=" ".join(caption.split()[0:max_length-1])
            del_idx.append(i)
        #caption=caption.replace('.', '')
        #caption = " ".join(caption.split())  # replace multiple spaces
        caption_data.set_value(i, 'caption', caption.lower())
    # delete captions if size is larger than max_length
    print("The number of captions before deletion: %d" % len(caption_data))
    caption_data = caption_data.drop(caption_data.index[del_idx])
    caption_data = caption_data.reset_index(drop=True)
    print ("The number of captions after deletion: %d" % len(caption_data))

    return caption_data

def _build_vocab(annotations, threshold=1):
    counter = Counter()
    max_len = 0
    for i, caption in enumerate(annotations['caption']):
        # words = caption.split(' ')  # caption contrains only lower-case words
        words=nltk.word_tokenize(caption)
        for w in words:
            counter[w] += 1

        # if len(caption.split(" ")) > max_len:
        #     max_len = len(caption.split(" "))
        if len(nltk.word_tokenize(caption)) > max_len:
            max_len = len(nltk.word_tokenize(caption))

    count_order = counter.most_common()

    count_order = OrderedDict(count_order)
    vocab = [word for word in count_order if count_order[word] >= threshold]
    print ('Filtered %d words to %d words with word count threshold %d.' % (len(count_order), len(vocab), threshold))

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
    idx = 3
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1
    print ("Max length of caption: ", max_len)

    process_vocab = {}
    process_vocab['vocab'] = vocab
    process_vocab['count_order'] = count_order
    return word_to_idx, process_vocab,max_len


def _build_caption_vector(annotations, word_to_idx, max_length=15):
    n_examples = len(annotations)
    captions = np.ndarray((n_examples, max_length + 2)).astype(np.int32)

    for i, caption in enumerate(annotations['caption']):
        # words = caption.split(" ")  # caption contrains only lower-case words
        words=nltk.word_tokenize(caption)
        cap_vec = []
        cap_vec.append(word_to_idx['<START>'])
        for word in words:
            if word in word_to_idx:
                cap_vec.append(word_to_idx[word])
        cap_vec.append(word_to_idx['<END>'])

        # pad short caption with the special null token '<NULL>' to make it fixed-size vector
        if len(cap_vec) < (max_length + 2):
            for j in range(max_length + 2 - len(cap_vec)):
                cap_vec.append(word_to_idx['<NULL>'])

        captions[i, :] = np.asarray(cap_vec)
    print ("Finished building caption vectors")
    return captions


def _build_file_names(annotations):
    image_file_names = []
    id_to_idx = {}
    idx = 0
    image_ids = annotations['report_id']
    file_names = annotations['images']
    for image_id, file_name in zip(image_ids, file_names):
        if not image_id in id_to_idx:
            id_to_idx[image_id] = idx
            image_file_names.append(file_name)
            idx += 1

    file_names = np.asarray(image_file_names)
    return file_names, id_to_idx


def _build_image_idxs(annotations, id_to_idx):
    image_idxs = np.ndarray(len(annotations), dtype=np.int32)
    image_ids = annotations['report_id']
    for i, image_id in enumerate(image_ids):
        image_idxs[i] = id_to_idx[image_id]
    return image_idxs


def main():
    # batch size for extracting feature vectors from vggnet.
    batch_size = 500
    # maximum length of caption(number of word). if caption is longer than max_length, deleted.
    max_length = 110
    # # if word occurs less than word_count_threshold in training dataset, the word index is special unknown token.
    word_count_threshold = 1
    ## Data Path
    
    data_path = '/data/Sample_test/'
    # # vgg model path
    vgg_model_path = './data/imagenet-vgg-verydeep-19.mat'
    #
    caption_file = 'data/annotations/captions_train2014.json'
    image_dir = 'image/%2014_resized/'
    #
    train_dataset = _process_caption_data(caption_data=data_path + 'train/train.json',max_length=max_length)



    test_dataset = _process_caption_data(caption_data=data_path + 'test/test.json',max_length=max_length)

    print ('Finished processing caption data')
    #
    save_pickle(train_dataset, data_path + 'train/train.annotations.pkl')

    save_pickle(test_dataset, data_path + 'test/test.annotations.pkl')

    #
    for split in ['train', 'test']:
        # for split in ['train','val']:
        annotations = load_pickle(data_path + '%s/%s.annotations.pkl' % (split, split))

        if split == 'train':
            word_to_idx, process_vocab,max_l = _build_vocab(annotations=annotations, threshold=word_count_threshold)
            save_pickle(word_to_idx, data_path + '%s/word_to_idx.pkl' % split)
            save_pickle(process_vocab, data_path + '%s/process-vocab.pkl' % split)
        #
        captions = _build_caption_vector(annotations=annotations, word_to_idx=word_to_idx, max_length=max_l)
        save_pickle(captions, data_path + '%s/%s.captions.pkl' % (split, split))
        #
        file_names, id_to_idx = _build_file_names(annotations)
        save_pickle(file_names, data_path + '%s/%s.file.names.pkl' % (split, split))

        image_idxs = _build_image_idxs(annotations, id_to_idx)
        save_pickle(image_idxs, data_path + '%s/%s.image.idxs.pkl' % (split, split))
        #
        # prepare reference captions to compute bleu scores later
        image_ids = {}
        feature_to_captions = {}
        i = -1
        for caption, image_id in zip(annotations['caption'], annotations['report_id']):
            if not image_id in image_ids:
                image_ids[image_id] = 0
                i += 1
                feature_to_captions[i] = []
            feature_to_captions[i].append(caption.lower() + ' .')
        save_pickle(feature_to_captions, data_path + '%s/%s.references.pkl' % (split, split))
        print ("Finished building %s caption dataset" % split)

    # extract conv5_3 feature vectors
    vggnet = Vgg19(vgg_model_path)
    vggnet.build()
    #model=pretrained_model()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for split in ['train', 'test']:
            anno_path = data_path + '%s/%s.annotations.pkl' % (split, split)
            save_path = data_path + '%s/%s.features.hkl' % (split, split)
            annotations = load_pickle(anno_path)
            image_path = annotations['images']
            n_examples = len(image_path)
            # ndarray to store image features from two images together
            all_feats = np.ndarray([n_examples, 196, 1024], dtype=np.float32)
            #all_feats = np.ndarray([n_examples, 196, 512], dtype=np.float32)
            i=0


            # for start, end in zip(range(0, n_examples, batch_size),
            for image_record in list(image_path):
                print(type(image_record))
                print(len(image_record))
                j = 0
                comb_map=np.ndarray([len(image_record), 196, 512], dtype=np.float32)
                for image in image_record:
                    #                     range(batch_size, n_examples + batch_size, batch_size)):
                    # image_batch_file = image_path[start:end]
                    image_batch_file = image
                    print(image_batch_file)
                    # # image_batch = np.array(map(lambda x: ndimage.imread(x+'.png', mode='RGB'), image_batch_file)).astype(np.float32)
                    # # image_batch = np.array(list(map(lambda x: ndimage.imread(x, mode='RGB'), image_batch_file))).astype(np.float32)
                    image_batch = np.expand_dims(np.array(ndimage.imread(image, mode='RGB').astype(np.float32)), axis=0)
                    # # print("shape:", image_batch.shape)
                    feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})
                    #feats = extract_features(model,image_batch_file)
                    #feats=featureVec_image(image_batch_file,model)
                    # feats=Main_hyper(image_batch)
                    # all_feats[start:end, :] = feats
                    comb_map[j,:]=feats
                    j+=1
                new_map=merge_feature_maps(comb_map)
                all_feats[i, :] = new_map
                #all_feats[i, :] = comb_map[0,:]
                i += 1
                # print ("Processed %d %s features.." % (end, split))
                print("Process %d %s features" % (i, split))
                # hickle.dump(all_feats, save_path)

            # use hickle to save huge feature vectors
            hickle.dump(all_feats, save_path)
            print ("Saved %s.." % (save_path))


if __name__ == "__main__":
    main()
