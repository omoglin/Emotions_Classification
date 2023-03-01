from Dataset import create_DataLoader
import Model_dropout
from Training_Loop import train_NN, accuracy, loss_visualization, metric_visualization


train_dir = "dataset_1/train"
val_dir = "dataset_1/test"
save_path = "Trained_Classification_Models"


train_DL = create_DataLoader(train_dir, val_dir, get_dataset='train', batch_size=64)
val_DL = create_DataLoader(train_dir, val_dir, get_dataset='validation', batch_size=64)

model = Model_dropout.ResNet50(img_channels=1)


losses_train_list, losses_val_list, metrics_train_list, metrics_val_list = train_NN(18, train_DL, val_DL,
                   model, save_path, accuracy, 'Adam')



'''
final_list = []
final_list.append('losses_train_list')
final_list.append(losses_train_list)
final_list.append('losses_val_list')
final_list.append(losses_val_list)
final_list.append('metrics_train_list')
final_list.append(metrics_train_list)
final_list.append('metrics_val_list')
final_list.append(metrics_val_list)


# open file in write mode
with open(r'mpl_charts_5.txt', 'w') as fp:
    for item in final_list:
        # write each item on a new line
        fp.write("%s\n" % item)
    #print('Done')
'''

loss_visualization(losses_train_list, losses_val_list)
metric_visualization(metrics_train_list, metrics_val_list)


'''
/ResNet152_3.pt & charts_1 & mpl_charts_1 = 0.0018 lr + 'Adam' + 40 epochs
/ResNet152_2.pt & charts_2 & mpl_charts_2 = 0.0018 lr + 'SGD' + 40 epochs
/ResNet50_1.pt & charts_3 & mpl_charts_3 = 0.0016 lr + 'Adam' + 45 epochs
/ResNet_50....pt & charts_4 & mpl_charts_4 = 0.00018 lr + 'Adam' + 190 epochs
/ResNet_50_dropout....pt & charts_5 & mpl_charts_5 = 0.00018 lr + 'Adam' + 170 epochs
'''