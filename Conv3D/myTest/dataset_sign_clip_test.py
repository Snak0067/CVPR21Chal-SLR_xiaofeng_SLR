from torchvision import transforms

from Conv3D.dataset_sign_clip import Sign_Isolated


if __name__ == '__main__':
    data_path = "../../data-prepare/data/frame/train_frame_data"
    label_train_path = "../../data-prepare/data/label/train_labels.csv"
    # sample number
    num_training = 2000
    sample_size = 128
    sample_duration = 32
    num_classes = 226
    transform = transforms.Compose([transforms.Resize([sample_size, sample_size]),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5], std=[0.5])])

    train_set = Sign_Isolated(data_path=data_path, label_path=label_train_path, frames=sample_duration,
                              num_classes=num_classes, train=True, transform=transform, sample_number=num_training)
