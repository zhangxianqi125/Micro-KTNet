import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
model.eval()
filename = 'dog.jpeg'

# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
#print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
#print(probabilities)
print(probabilities.argmax())


def get_pretrained_microscopynet_url(encoder, encoder_weights):
    """Get the url to download the specified pretrained encoder.

    Args:
        encoder (str): pretrained encoder model name (e.g. resnet50)
        encoder_weights (str): pretraining dataset, either 'microscopynet' or 
            'imagenet-microscopynet' with the latter indicating the encoder
            was first pretrained on imagenet and then finetuned on microscopynet

    Returns:
        str: url to download the pretrained model
    """
    url_base = 'https://nasa-public-data.s3.amazonaws.com/microscopy_segmentation_models/'
    url_end = '_v1.0.pth.tar'
    return url_base + f'{encoder}_pretrained_{encoder_weights}' + url_end

import torch.utils.model_zoo as model_zoo

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
url = get_pretrained_microscopynet_url('resnet50', 'microscopynet')
model.load_state_dict(model_zoo.load_url(url))
model.eval()  # <---- this is the MicroscopyNet model


filename = 'npg.png'

input_image = Image.open(filename).convert('RGB')
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)


probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities.argmax())