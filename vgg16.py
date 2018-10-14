import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import decode_predictions
from keras_applications.vgg16 import preprocess_input
from keras_preprocessing import image


def _load_image(img_path):
    """
    function to load and to pre-process the image file
    :param img_path: path to the image file
    :return: img
    """
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def _get_predictions_vgg16(imgs):
    """
    function to get predictions with vgg16 model from keras
    :param imgs: list of images to be predicted
    """
    vgg16_weights = '../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    _model = VGG16(weights=vgg16_weights)

    f, ax = plt.subplots(1, 4)
    f.set_size_inches(80, 48)
    for i in range(len(imgs)):
        ax[i].imshow(Image.open(imgs[i]).resize((200, 200), Image.ANTIALIAS))
    plt.show()

    f, axes = plt.subplots(1, len(imgs))
    f.set_size_inches(80, 20)
    for i, img_path in enumerate(imgs):
        img = _load_image(img_path)
        preds = decode_predictions(_model.predict(img), top=3)[0]
        preds_1 = [c[1] for c in preds]
        preds_p = [c[2] for c in preds]
        _b = sns.barplot(y=preds_1, x=preds_p, color='gray', ax=axes[i])
        f.tight_layout()
