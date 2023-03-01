import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def create_optimizer(model, opt_scheme, lr=0.00018):
    if opt_scheme == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr)
    elif opt_scheme == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr)
    return optimizer
#opt = create_optimizer(model)


# CREATE ACCURACY METRIC
def accuracy(model_output, yb):
    probabilities = F.softmax(model_output, dim=1)
    #print(f"Shape of the probabilities' tensor: {probabilities.shape}")
    #print(torch.sum(probabilities, dim=1))

    max_values, max_labels = torch.max(probabilities, dim=1)
    acc_score = torch.sum(max_labels == yb) / len(max_labels)
    return acc_score

#accuracy(model_output, yb)


def training_scheme(xb, yb, model, metric, optimizer=None):
    # GPU STUFF !!!
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xb, yb = xb.to(device), yb.to(device)
    model = model.to(device)
    
    # xb = image data; yb = real labels
    output = model(xb)
    loss = F.cross_entropy(output, yb)

    if optimizer is not None:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    metric_results = metric(output, yb)
    
    #print(f"{metric.__name__} score: {metric_results}")
    #print(f"loss score: {loss.item()}")
    return loss.item(), metric_results.item()

#training_scheme(xb, yb, model, accuracy)



def evaluate(eval_DL, model, metric):

    with torch.no_grad():
        loss_val_list, metric_val_list = [], []

        for i, (xb, yb) in enumerate(eval_DL):
            loss_val_score, metric_val_score = training_scheme(xb, yb, model, metric)
            loss_val_list.append(loss_val_score)
            metric_val_list.append(metric_val_score)

            #TODO: Perhaps delete later?
            if (i + 1) % 50 == 0:
                print(f"Batch pass nr: {i + 1}")
        print("")

        final_val_loss_score = sum(loss_val_list) / len(loss_val_list)
        final_val_metric_score = sum(metric_val_list) / len(metric_val_list)

    return final_val_loss_score, final_val_metric_score

#evaluate(val_DL, model, accuracy)


# save_path = "Trained_Classification_Models"
def save_model(model, save_folder_path):
    torch.save(model, save_folder_path)


def train_NN(num_epochs, train_DL, eval_DL, model, save_folder_path, metric, opt_scheme='SGD'):
    
    optimizer = create_optimizer(model, opt_scheme=opt_scheme)
    
    model.eval()
    print("passing the training set through the model for the initial evaluation...")
    initial_train_loss_score, initial_train_metric_score = evaluate(train_DL, model, metric)
    print("passing the validation set through the model for the initial evaluation...")
    initial_val_loss_score, initial_val_metric_score = evaluate(eval_DL, model, metric)

    print(f"Initial Training Loss Score: {initial_train_loss_score}, Initial Training {metric.__name__} Score: {initial_train_metric_score}")
    print(f"Initial Validation Loss Score: {initial_val_loss_score}, Initial Validation {metric.__name__} Score: {initial_val_metric_score}")
    print("")

    losses_train_list, losses_val_list, metrics_train_list, metrics_val_list = [], [], [], []
    for epoch in range(num_epochs):
        model.train()
        loss_train_list, metric_train_list = [], []
        for i, (xb, yb) in enumerate(train_DL):
            loss_train_score, metric_train_score = training_scheme(xb, yb, model, metric, optimizer)
            loss_train_list.append(loss_train_score)
            metric_train_list.append(metric_train_score)
            if (i+1) % 50 == 0:
                print(f"Number of batch passes through the entire iteration: {i + 1}")
        print("")

        final_train_loss_score = sum(loss_train_list) / len(loss_train_list)
        final_train_metric_score = sum(metric_train_list) / len(metric_train_list)

        model.eval()
        final_val_loss_score, final_val_metric_score = evaluate(eval_DL, model, metric)

        losses_train_list.append(final_train_loss_score)
        losses_val_list.append(final_val_loss_score)
        metrics_train_list.append(final_train_metric_score)
        metrics_val_list.append(final_val_metric_score)

        if (epoch + 1) % 20 == 0:
            save_model(model, save_folder_path + "/ResNet50_dropout" + f"_epoch_{epoch + 1}.pt")

        print(f"Iteration / Num Iterations: {epoch + 1} / {num_epochs}, Train Loss: {final_train_loss_score},"
              f"Train {metric.__name__}: {final_train_metric_score}, Validation Loss: {final_val_loss_score}, "
              f"Validation {metric.__name__}: {final_val_metric_score}")
        print("")
        print("")

    save_model(model, save_folder_path + "/ResNet50_dropout_end_score.pt")

    return losses_train_list, losses_val_list, metrics_train_list, metrics_val_list


#losses_train_list, losses_val_list, metrics_train_list, metrics_val_list = train_NN(6, train_DL, val_DL, model, accuracy, opt)



def loss_visualization(losses_train_list, losses_val_list):
    plt.figure(figsize=(8, 8))
    plt.title("Loss Score Visualization")
    plt.plot(list(range(len(losses_train_list))), losses_train_list, linewidth=1.25, c="darkred", label="Training Loss")
    plt.plot(list(range(len(losses_val_list))), losses_val_list, linewidth=1.25, c="darkblue", label="Validation Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss Score")
    plt.legend()
    plt.tight_layout()
    plt.show()

#loss_visualization(losses_train_list, losses_val_list)


def metric_visualization(metrics_train_list, metrics_val_list):
    plt.figure(figsize=(8, 8))
    plt.title("Accuracy Score Visualization")
    plt.plot(list(range(len(metrics_train_list))), metrics_train_list, linewidth=1.25, c="darkred", label="Training Accuracy")
    plt.plot(list(range(len(metrics_val_list))), metrics_val_list, linewidth=1.25, c="darkblue", label="Validation Accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy Score")
    plt.legend()
    plt.tight_layout()
    plt.show()

#metric_visualization(metrics_train_list, metrics_val_list)
