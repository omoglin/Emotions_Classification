from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms


# train_dir = "dataset_1/train"
# val_dir = "dataset_1/test"


# creating a dataset for mean and standard deviation extraction
def create_simple_dataset(train_directory):
    Get_Mean_Std_transform_function = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    # print(len(train_dataset_Get_Mean_STD))
    return ImageFolder(train_directory, transform=Get_Mean_Std_transform_function)



# dataset = create_simple_dataset
def calculate_stats(train_directory):
    dataset = create_simple_dataset(train_directory)

    mean_list, std_list = [], []
    for img_number, (img, label) in enumerate(dataset):
        mean_list.append(img.mean().item())
        std_list.append(img.std().item())
        # print(f"Image Number: {img_number + 1}")
    # print("")
    mean_avg = sum(mean_list) / len(mean_list)
    std_avg = sum(std_list) / len(std_list)

    # print(f"Dataset Average Mean: {mean_avg}")
    # print(f"Dataset Average Standard Deviation: {std_avg}")
    # print("")

    return mean_avg, std_avg



def create_transform_functions(train_directory):
    mean, std = calculate_stats(train_directory)

    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomCrop(48, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((mean, ), (std, ))
    ])

    val_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((mean, ), (std, ))
    ])
    return train_transforms, val_transforms



def create_datasets(train_directory, validation_directory, get_dataset):
    train_transforms, val_transforms = create_transform_functions(train_directory)
    train_dataset = ImageFolder(train_directory, transform=train_transforms)
    val_dataset = ImageFolder(validation_directory, transform=val_transforms)

    if get_dataset == 'train':
        return train_dataset
    elif get_dataset == 'validation':
        return val_dataset



def create_DataLoader(train_directory, validation_directory, get_dataset='train', batch_size=64):
    dataset = create_datasets(train_directory, validation_directory, get_dataset)

    # OVERSAMPLING METHOD
    class_weights = [435/3981, 1, 435/4090, 435/7203, 435/4952, 435/4826, 435/3163]
    sample_weights = [0]*len(dataset)
    for idx, (data, label) in enumerate(dataset):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight
    sampler = WeightedRandomSampler(sample_weights, num_samples = len(sample_weights), replacement=True)

    return DataLoader(dataset, batch_size, sampler=sampler)

