import anchor_base
import numpy as np
import sklearn
import skimage
from skimage import color
import torch


class AnchorImage(object):
    """Purpose: This class is used to generate an explanation for an image"""

    def __init__(self, distribution_path=None,
                 transform_img_fn=None, n=1000, dummys=None, white=None,
                 segmentation_fn=None):
        """
        Params: 
            distribution_path: path to a directory containing images.
            transform_img_fn: a function that takes a path to an image and returns a transformed image.
            n: number of images to load from distribution_path.
            dummys: a list of images to use as dummy examples.
            white: a color to use for the dummy examples.
            segmentation_fn: a function that takes an image and returns a segmentation.

        Params of Initialization:
            - hide: a boolean that determines whether to use dummy examples.
            - white: a color to use for the dummy examples.
            - segmentation: a function that takes an image and returns a segmentation.
            - dummys: a list of images to use as dummy examples.
        """
        self.hide = True
        self.white = white
        if segmentation_fn is None:
            from skimage.segmentation import quickshift

            def segmentation_fn(x): return quickshift(x, kernel_size=4,  # noqa
                                                   max_dist=200, ratio=0.2)
        self.segmentation = segmentation_fn
        if dummys is not None:
            self.hide = False
            self.dummys = dummys
        elif distribution_path:
            self.hide = False
            import os
            import skimage

            if not transform_img_fn:
                def transform_img(path):
                    img = skimage.io.imread(path)
                    short_egde = min(img.shape[:2])
                    yy = int((img.shape[0] - short_egde) / 2)
                    xx = int((img.shape[1] - short_egde) / 2)
                    crop_img = img[yy: yy + short_egde, xx: xx + short_egde]
                    return skimage.transform.resize(crop_img, (224, 224))

                def transform_imgs(paths):
                    out = []
                    for i, path in enumerate(paths):
                        if i % 100 == 0:
                            print(i)
                        out.append(transform_img(path))
                    return out
                transform_img_fn = transform_imgs
            all_files = os.listdir(distribution_path)
            all_files = np.random.choice(
                all_files, size=min(n, len(all_files)), replace=False)
            paths = [os.path.join(distribution_path, f) for f in all_files]
            self.dummys = transform_img_fn(paths)

    def get_sample_fn(self, image, classifier_fn, lime=False):
        """
        Params: 
            - image: the image to be explained.
            - classifier_fn: a function that takes a numpy array and outputs prediction probabilities.
            - lime: a boolean that determines whether to use LIME.
        Purpose: This function generates a sample function for the image.
        Outputs: A tuple containing the segments and the sample function.
        """
        import copy
        # segments = slic(image, n_segments=100, compactness=20)
        segments = self.segmentation(image)
        fudged_image = image.copy()
        ngu = np.unique(segments)
        for x in np.unique(segments):
            fudged_image[segments == x] = (np.mean(image[segments == x][:, 0]),
                                           np.mean(image[segments == x][:, 1]),
                                           np.mean(image[segments == x][:, 2]))
        if self.white is not None:
            fudged_image[:] = self.white
        features = list(np.unique(segments))
        n_features = len(features)

        true_label = np.argmax(classifier_fn(np.expand_dims(image, 0))[0])
        print('True pred', true_label)

        def lime_sample_fn(num_samples, batch_size=50):
            """
            Params: 
                - num_samples: the number of samples to generate.
                - batch_size: the size of the batch to use when generating samples.
            Purpose: This function generates samples for LIME.
            Outputs: A tuple containing the samples and the labels.
            """
            # data = np.random.randint(0, 2, num_samples * n_features).reshape(
            #     (num_samples, n_features))
            data = np.zeros((num_samples, n_features))
            labels = []
            imgs = []
            sizes = np.random.randint(0, n_features, num_samples)
            all_features = range(n_features)
            # for row in data:
            for i, size in enumerate(sizes):
                row = np.ones(n_features)
                chosen = np.random.choice(all_features, size)
                # print chosen, size,
                row[chosen] = 0
                data[i] = row
                # print row
                temp = copy.deepcopy(image)
                zeros = np.where(row == 0)[0]
                mask = np.zeros(segments.shape).astype(bool)
                for z in zeros:
                    mask[segments == z] = True
                temp[mask] = fudged_image[mask]
                imgs.append(temp)
                if len(imgs) == batch_size:
                    preds = classifier_fn(np.array(imgs))
                    labels.extend(preds)
                    imgs = []
            if len(imgs) > 0:
                preds = classifier_fn(np.array(imgs))
                labels.extend(preds)
            # return imgs, np.array(labels)
            return data, np.array(labels)

        if lime:
            return segments, lime_sample_fn

        def sample_fn_dummy(present, num_samples, compute_labels=True):
            """
            Params:
                - present (list): a list of features that are present.
                - num_samples (int): the number of samples to generate.
                - compute_labels (bool): a boolean that determines whether to compute labels.
            Purpose: This function generates samples for the image.
            Outputs: A tuple containing the raw data, the data, and the labels.
            """
            if not compute_labels:
                data = np.random.randint(
                    0, 2, num_samples * n_features).reshape(
                        (num_samples, n_features))
                data[:, present] = 1
                return [], data, []
            data = np.zeros((num_samples, n_features))
            # data = np.random.randint(0, 2, num_samples * n_features).reshape(
            #     (num_samples, n_features))
            if len(present) < 5:
                data = np.random.choice(
                    [0, 1], num_samples * n_features, p=[.8, .2]).reshape(
                        (num_samples, n_features))
            data[:, present] = 1
            chosen = np.random.choice(range(len(self.dummys)), data.shape[0],
                                      replace=True)
            labels = []
            imgs = []
            for d, r in zip(data, chosen):
                temp = copy.deepcopy(image)
                zeros = np.where(d == 0)[0] # return index
                mask = np.zeros(segments.shape).astype(bool)
                for z in zeros:
                    mask[segments == z] = True
                if self.white:
                    temp[mask] = 1
                else:
                    temp[mask] = self.dummys[r][mask]
                imgs.append(temp)
                # pred = np.argmax(classifier_fn(temp.to_nn())[0])
                # print self.class_names[pred]
                # labels.append(int(pred == true_label))
            # import time
            # a = time.time()
            imgs = np.array(imgs)
            preds = classifier_fn(imgs)
            # print (time.time() - a) / preds.shape[0]
            imgs = []
            preds_max = np.argmax(preds, axis=1)
            labels = (preds_max == true_label).astype(int)
            raw_data = np.hstack((data, chosen.reshape(-1, 1)))
            return raw_data, data, np.array(labels)

        def sample_fn(present, num_samples, compute_labels=True):
            """
            Params: 
                - present (list): a list of features that are present.
                - num_samples (int): the number of samples to generate.
                - compute_labels (bool): a boolean that determines whether to compute labels.
            Purpose: This function generates samples for the image.
            Outputs: A tuple containing the raw data, the data, and the labels.
            """

            # TODO: I'm sampling in this different way because the way we were
            # sampling confounds size of the document with feature presence
            # (larger documents are more likely to have features present)
            data = np.random.randint(0, 2, num_samples * n_features).reshape(
                (num_samples, n_features))
            data[:, present] = 1
            if not compute_labels:
                return [], data, []
            imgs = []
            for row in data:
                temp = copy.deepcopy(image)
                zeros = np.where(row == 0)[0]
                mask = np.zeros(segments.shape).astype(bool)
                for z in zeros:
                    mask[segments == z] = True
                temp[mask] = fudged_image[mask]
                imgs.append(temp)
            preds = classifier_fn(np.array(imgs))
            preds_max = np.argmax(preds, axis=1)
            labels = (preds_max == true_label).astype(int)
            # raw_data = imgs
            raw_data = data
            return raw_data, data, labels

        sample = sample_fn if self.hide else sample_fn_dummy
        return segments, sample

    def explain_instance(self, image, classifier_fn, threshold=0.95,
                         delta=0.1, tau=0.15, batch_size=100,
                         **kwargs):
        """
        Params: 
            - image: the image to be explained.
            - classifier_fn: a function that takes a numpy array and outputs prediction probabilities.
            - threshold: the desired confidence level for the anchor.
            - delta: the desired coverage level for the anchor.
            - tau: the desired precision level for the anchor.
            - batch_size: the size of the batch to use when generating anchors.
            - **kwargs: additional arguments to pass to the anchor_beam function.
        Purpose: This function generates an explanation for the image.
        Outputs: A tuple containing the segments and the explanation.
        """
        segments, sample = self.get_sample_fn(image, classifier_fn)
        exp = anchor_base.AnchorBaseBeam.anchor_beam(
            sample, delta=delta, epsilon=tau, batch_size=batch_size,
            desired_confidence=threshold, **kwargs)
        return segments, self.get_exp_from_hoeffding(image, exp)

    def get_exp_from_hoeffding(self, image, hoeffding_exp):
        """
        bla
        """
        ret = []

        features = hoeffding_exp['feature']
        means = hoeffding_exp['mean']
        if 'negatives' not in hoeffding_exp:
            negatives_ = [np.array([]) for x in features]
        else:
            negatives_ = hoeffding_exp['negatives']
        for f, mean, negatives in zip(features, means, negatives_):
            train_support = 0
            name = ''
            if negatives.shape[0] > 0:
                unique_negatives = np.vstack({
                    tuple(row) for row in negatives})
                distances = sklearn.metrics.pairwise_distances(
                    np.ones((1, negatives.shape[1])),
                    unique_negatives)
                negative_arrays = (unique_negatives
                                   [np.argsort(distances)[0][:4]])
                negatives = []
                for n in negative_arrays:
                    negatives.append(n)
            else:
                negatives = []
            ret.append((f, name, mean, negatives, train_support))
        return ret


def transform_img_fast(path):
    """Purpose: Crop and resize the image to 299x299 for InceptionV3"""
    img = skimage.io.imread(path)
    if len(img.shape) != 3:
        img = skimage.color.gray2rgb(img)
    if img.shape[2] == 4:
        img = color.rgba2rgb(img)
    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    crop_img = img[yy: yy + short_egde, xx: xx + short_egde]
    return (skimage.transform.resize(crop_img, (299, 299)) - 0.5) * 2


def transform_img_fn_fast(paths):
    """Purpose: Transform a list of image paths to a numpy array of images"""
    out = []
    for i, path in enumerate(paths):
        if i % 100 == 0:
            print(i)
        out.append(transform_img_fast(path))
    return np.array(out)
#     return np.array([transform_img_fast(path) for path in paths])

def predict(images):
    """Purpose: Predict the class of the image using the InceptionV3 model"""
    images = images.transpose((0, 3, 1, 2)) # Shape in Pytorch: NxCxHxW (N: Number of images, C: Number of channels, H: Height, W: Width)
    input_tensor = torch.FloatTensor(images)
    if torch.cuda.is_available():
        input_tensor = input_tensor.to('cuda')
    with torch.no_grad():
        output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    return probabilities.cpu().numpy()

model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
model.eval()

if torch.cuda.is_available():
    print("GPU is available")
    model.to('cuda')
    
images = transform_img_fn_fast(['F:/References/GraduateThesis-Anchor/anchor-master/notebooks/nick.png'])
# Initialize the explainer
explainer = AnchorImage('F:/References/GraduateThesis-Anchor/anchor-master/dataset',
                        transform_img_fn=transform_img_fn_fast, n=20)
segments, exp = explainer.explain_instance(images[0], predict, threshold=0.95, batch_size=50,
                                           tau=0.20, verbose=True, min_shared_samples=200, beam_size=2)
