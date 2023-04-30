import torch
import torchvision
import numpy as np
import random
import math
import matplotlib.pyplot as plt
# Data loading and visualization imports
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from matplotlib.pyplot import imshow, figure, subplots

# Model loading
from model.erfnet import Net as ERFNet
from model.lcnet import Net as LCNet


# to cuda or not to cuda
if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

# utils
# Function used to map a 1xWxH class tensor to a 3xWxH color image
def color_lanes(image, classes, i, color, HEIGHT, WIDTH):
    buffer_c1 = np.zeros((HEIGHT, WIDTH))
    buffer_c1[classes == i] = color[0]   
    image[:, :, 0] += buffer_c1

    buffer_c2 = np.zeros((HEIGHT, WIDTH))
    buffer_c2[classes == i] = color[1]   
    image[:, :, 1] += buffer_c2

    buffer_c3 = np.zeros((HEIGHT, WIDTH))
    buffer_c3[classes == i] = color[2]   
    image[:, :, 2] += buffer_c3
    return image

def blend(image_orig, image_classes): 
	image_classes = image_classes.astype(np.uint8)
	mask = np.zeros(image_classes.shape)

	mask[image_classes.nonzero()] = 255
	mask = mask[:, :, 0]
	mask = Image.fromarray(mask.astype(np.uint8))
	image_classes = Image.fromarray(image_classes)

	image_orig.paste(image_classes, None, mask)
	return image_orig

DESCRIPTOR_SIZE = 64

NUM_CLASSES_SEGMENTATION = 5

NUM_CLASSES_CLASSIFICATION = 3

HEIGHT = 360
WIDTH = 640


image = []
i = Image.open('./Datasets/test.jpg')
image.append(i)
i = Image.open('./Datasets/solidWhiteCurve.jpg')
image.append(i)
i = Image.open('./Datasets/solidWhiteRight.jpg')
image.append(i)
i = Image.open('./Datasets/solidYellowCurve.jpg')
image.append(i)
i = Image.open('./Datasets/solidYellowCurve2.jpg')
image.append(i)
i = Image.open('./Datasets/solidYellowLeft.jpg')
image.append(i)
i = Image.open('./Datasets/whiteCarLaneSwitch.jpg')
image.append(i)

# ipynb visualization
for im in image:
    plt.imshow(im)
    plt.show()
    imshow(np.asarray(im))

    im = im.resize((WIDTH, HEIGHT))

    im_tensor = ToTensor()(im)
    im_tensor = im_tensor.unsqueeze(0)

    # Creating CNNs and loading pretrained models
    segmentation_network = ERFNet(NUM_CLASSES_SEGMENTATION)
    classification_network = LCNet(NUM_CLASSES_CLASSIFICATION, DESCRIPTOR_SIZE, DESCRIPTOR_SIZE)

    segmentation_network.load_state_dict(torch.load('./erfnet_tusimple.pth', map_location = map_location))
    model_path = 'pretrained/classification_{}_{}class.pth'.format(DESCRIPTOR_SIZE, NUM_CLASSES_CLASSIFICATION)
    classification_network.load_state_dict(torch.load(model_path, map_location = map_location))

    segmentation_network = segmentation_network.eval()
    classification_network = classification_network.eval()

    if torch.cuda.is_available():
        segmentation_network = segmentation_network.cuda()
        classification_network = classification_network.cuda()

    # Inference on instance segmentation
    if torch.cuda.is_available():
        im_tensor = im_tensor.cuda()

    out_segmentation = segmentation_network(im_tensor)
    out_segmentation = out_segmentation.max(dim=1)[1]


    out_segmentation_np = out_segmentation.cpu().numpy()[0]
    out_segmentation_viz = np.zeros((HEIGHT, WIDTH, 3))

    for i in range(1, NUM_CLASSES_SEGMENTATION):
        rand_c1 = np.random.randint(1, 255)
        rand_c2 = np.random.randint(1, 255)    
        rand_c3 = np.random.randint(1, 255)
        out_segmentation_viz = color_lanes(
            out_segmentation_viz, out_segmentation_np, 
            i, (rand_c1, rand_c2, rand_c3), HEIGHT, WIDTH)

    # Display the image
    plt.imshow(out_segmentation_viz)
    plt.show()


    def extract_descriptors(label, image):
        # avoids problems in the sampling
        eps = 0.01
        
        # The labels indices are not contiguous e.g. we can have index 1, 2, and 4 in an image
        # For this reason, we should construct the descriptor array sequentially
        inputs = torch.zeros(0, 3, DESCRIPTOR_SIZE, DESCRIPTOR_SIZE)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            
        # This is needed to keep track of the lane we are classifying
        mapper = {}
        classifier_index = 0
        
        # Iterating over all the possible lanes ids
        for i in range(1, NUM_CLASSES_SEGMENTATION):
            # This extracts all the points belonging to a lane with id = i
            single_lane = label.eq(i).view(-1).nonzero().squeeze()
            
            # As they could be not continuous, skip the ones that have no points
            if single_lane.numel() == 0 or len(single_lane.size()) == 0:
                continue
            
            # Points to sample to fill a squared desciptor
            sample = torch.zeros(DESCRIPTOR_SIZE * DESCRIPTOR_SIZE)
            if torch.cuda.is_available():
                sample = sample.cuda()
                
            sample = sample.uniform_(0, single_lane.size()[0] - eps).long()
            sample, _ = sample.sort()
            
            # These are the points of the lane to select
            points = torch.index_select(single_lane, 0, sample)
            
            # First, we view the image as a set of ordered points
            descriptor = image.squeeze().view(3, -1)
            
            # Then we select only the previously extracted values
            descriptor = torch.index_select(descriptor, 1, points)
            
            # Reshape to get the actual image
            descriptor = descriptor.view(3, DESCRIPTOR_SIZE, DESCRIPTOR_SIZE)
            descriptor = descriptor.unsqueeze(0)
            
            # Concatenate to get a batch that can be processed from the other network
            inputs = torch.cat((inputs, descriptor), 0)
            
            # Track the indices
            mapper[classifier_index] = i
            classifier_index += 1
            
        return inputs, mapper


    descriptors, index_map = extract_descriptors(out_segmentation, im_tensor)

    GRID_SIZE = 2
    _, fig = subplots(GRID_SIZE, GRID_SIZE)

    for i in range(0, descriptors.size(0)):
        desc = descriptors[i].cpu()

        desc = ToPILImage()(desc)
        row = math.floor((i / GRID_SIZE))
        col = i % GRID_SIZE

        fig[row, col].imshow(np.asarray(desc))
    plt.show()

    # Inference on descriptors
    classes = classification_network(descriptors).max(1)[1]
    print(index_map)
    print(classes)

    # Class visualization
    out_classification_viz = np.zeros((HEIGHT, WIDTH, 3))

    for i, lane_index in index_map.items():
        if classes[i] == 0: # Continuous
            color = (255, 0, 0)
        elif classes[i] == 1: # Dashed
            color = (0, 255, 0)
        elif classes[i] == 2: # Double-dashed
            color = (0, 0, 255)
        else:
            raise
        out_classification_viz[out_segmentation_np == lane_index] = color

    imshow(out_classification_viz.astype(np.uint8))
    plt.show()
