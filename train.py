import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader
from MIX import FMixer
from torch.utils.tensorboard import SummaryWriter
import torch.onnx
from MIX import EnMixer



def get_data(file_name):
    
    # 定义空列表，用于存储数据
    data_list = []
    label_list = []
    
    with open(file_name, "r") as f1:
        for line in f1:

            list = eval(line)
            lst_data = list[2:42]
            lst_label = list[:2]

            data_list.append(lst_data)
            label_list.append(lst_label)
            
          
    f1.close()
    data_tensor = torch.tensor(data_list)
    label_tensor = torch.tensor(label_list)

    data_tensor = data_tensor.to(torch.float32)
    label_tensor = label_tensor.to(torch.float32)
    print(data_tensor.size())
    print(label_tensor.size())
    
    return data_tensor, label_tensor


def train():
    
    n_splits = 5
    batch_size = 8192
    models = []
    writers = []
    seeds = [32, 123, 456, 789, 1024]
    device = torch.device("cuda:0")
    loss_fn = nn.MSELoss()

    print('loading')
    X, Y = get_data("./data/setcover_train_and_dev_data.txt")
    print('loaded')

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=32)

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False)
        seed = seeds.pop(0) 
        torch.manual_seed(seed) 

        model = FMixer(n_block=2)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        writer = SummaryWriter(log_dir=f"logs/liu_mix_model_{i}")

        writers.append(writer)
        best_loss = float("inf")
        best_state_dict = None

        epochs = 200
        total_batch = 0
        last_improve = 0

        for epoch in range(epochs):
            for inputs, targets in train_loader:

                model.train()
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
                optimizer.step()

                if total_batch % 100 == 0 and total_batch != 0:
                    model.eval()
                    test_loss = 0.0

                    with torch.no_grad():
                        for inputs, targets in test_loader:
                            inputs = inputs.to(device)
                            targets = targets.to(device)
                            outputs = model(inputs)
                            Tloss = loss_fn(outputs, targets)

                            test_loss += Tloss.item()
                    test_loss /= len(test_loader)

                    if test_loss < best_loss:
                        best_loss = test_loss
                        torch.save(model.state_dict(), "./liu_setcover_best_model_EnMix_"+str(i)+".ckpt")
                        last_improve = total_batch

                    print(f"Epoch {epoch + 1}, total_batch: {total_batch}, Train Loss: {loss.item():>5.3}, Test Loss: {test_loss::>5.3}, last_improve: {last_improve}")
                    writer.add_scalar("loss/liu_train_loss", loss.item(), total_batch)
                    writer.add_scalar("loss/liu_test_loss", test_loss, total_batch)
                total_batch += 1

            scheduler.step()

    for writer in writers:
        writer.close()

if __name__ == "__main__":

    train()

    device = torch.device("cpu")
    model = EnMixer()
    model.eval()
    example_input = torch.tensor([1.008382, 1.201734, 0.0, 1.0, 0.0, -0.364382, 0.0, 0.0, 0.0, 2.519651, 0.0, 0.050779, 1.165065, 0.165065, 0.0, 0.0, 0.0, 0.01, 0.0, 1.0, 1.008382, 1.201734, 0.0, 1.0, 0.0, -0.364382, 0.0, 0.0, 0.0, 2.519651, 0.0, 0.050779, 1.165065, 0.165065, 0.0, 0.0, 0.0, 0.01, 0.0, 1.0])
    input_names = ["input"]
    output_names = ["output"]
    # OUTPUT 
    onnx_path = "setcover_liu_contrast_enmix.onnx"
    torch.onnx.export(model, example_input, onnx_path,input_names=input_names,output_names=output_names, verbose=True, do_constant_folding=True)
