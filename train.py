from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import load_coco_data
import tensorflow as tf
def execute_Two_images(path):

    data = load_coco_data(data_path=path, split='train')
    word_to_idx = data['word_to_idx']
    # load val dataset to print out bleu scores every epoch
    val_data = load_coco_data(data_path=path, split='val')
    # load test dataset to print out bleu scores
    test_data = load_coco_data(data_path=path, split='test')
    model = CaptionGenerator(word_to_idx, dim_feature=[196, 1024], dim_embed=512,
                             dim_hidden=1024, n_time_step=14, prev2out=True,
                             ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

    solver = CaptioningSolver(model, data, val_data, test_data, n_epochs=60, batch_size=16, update_rule='adam',
                              learning_rate=0.001, print_every=100, save_every=10, image_path='./image/',
                              pretrained_model=None, model_path='model/lstm/', test_model='model/lstm/model-60',
                              print_bleu=True, log_path='log/', path=path)
    solver.train()

def execute(path):

    data = load_coco_data(data_path=path, split='train')
    word_to_idx = data['word_to_idx']
    # load val dataset to print out bleu scores every epoch
    val_data = load_coco_data(data_path=path, split='val')
    # load test dataset to print out bleu scores
    test_data = load_coco_data(data_path=path, split='test')
    model = CaptionGenerator(word_to_idx, dim_feature=[196,1024], dim_embed=512,
                             dim_hidden=1024, n_time_step=101, prev2out=True,
                             ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

    solver = CaptioningSolver(model, data, val_data, test_data, n_epochs=60, batch_size=16, update_rule='adam',
                              learning_rate=0.0001, print_every=100, save_every=10, image_path='./image/',
                              pretrained_model=None, model_path='model/lstm/', test_model='model/lstm/model-60',
                              print_bleu=True, log_path='log/', path=path)

    solver.train()
def main():

    print(tf.test.gpu_device_name())
    # load train dataset
    data_path='/media/zaheer/Data/Image_Text_Datasets/IU_Xray/latest/Two_Images/word/impression_first/Sample'
    #data_path = '/media/zaheer/Data/Image_Text_Datasets/IU_Xray/latest/two_images_split/sent/Sample'
    Samples=['_weight']

    for sample in Samples:
        print("Sample:",sample)
        path = data_path + str(sample) + '/'
        execute(path)





if __name__ == "__main__":
    main()
