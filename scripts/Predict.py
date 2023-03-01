import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2



def create_transform_function(mean_avg, std_avg):
    transform = transforms.Compose([
        transforms.Resize(48),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((mean_avg, ), (std_avg, ))
    ])
    return transform

# transform_function = create_transform_function(0.5076887596402362, 0.21222973403372375)



# FINAL FUNCTION TO PREDICT
def predict(path_to_image, model, classes, print_text=False, put_text=True, colour = (255, 127, 127)):

    image = Image.open(path_to_image)

    transform_fn = create_transform_function(0.5076887596402362, 0.21222973403372375)
    image_transformed = transform_fn(image)

    # transformations necessary to visualize the image
    image_array = np.array(image)

    #print(image_array.shape[0])
    #print(image_array.shape[1])
    img_size = image_array.size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_transformed = image_transformed.to(device)
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        output = model(image_transformed.unsqueeze(0))
    probabilities = F.softmax(output, dim=1)
    max_value, max_label = torch.max(probabilities, dim=1)

    if print_text:
        print(f'Predicted Emotion: {classes[max_label].upper()}, With The Probability Of: '
              f'{ str(round(max_value.item() * 100, 2)) + "%" }')
        print("")

    if put_text:
        axis_1 = int(image_array.shape[0]/25)
        axis_2 = int(image_array.shape[1]/25)
        font_style = cv2.FONT_HERSHEY_SIMPLEX
        size = img_size / 3000000 # (img_size * 2)
        #colour = (255, 127, 127)   # already made as a function's argument
        thicnkness = 2

        cv2.putText(image_array, f'Predicted Emotion: {classes[max_label].upper()}',
                    (axis_1, axis_2), font_style, size, colour, thickness=thicnkness)

        cv2.putText(image_array, f'With The Probability Of: {str(round(max_value.item() * 100, 2)) + "%"}',
                    (axis_1, axis_2*2), font_style, size, colour, thickness=thicnkness)

    image_converted = Image.fromarray(image_array)
    image_converted.show()


