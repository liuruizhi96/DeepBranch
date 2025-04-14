import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
  """Residual block of FMixer."""

  def __init__(self):
    super().__init__()
    self.dropout = nn.Dropout(p=0.2)
    # node Linear
    self.node_norm = nn.BatchNorm1d(2)
    self.node_linear1 = nn.Linear(2, 128)
    self.node_linear = nn.Linear(128, 128)
    self.node_linear2 = nn.Linear(128, 2)

    # Feature Linear
    self.feature_norm = nn.BatchNorm1d(2)
    self.feature_linear1 = nn.Linear(20, 128)
    self.feature_linear = nn.Linear(128, 128)
    self.feature_linear2 = nn.Linear(128, 20)

  def forward(self, inputs):
    
    x = self.node_norm(inputs) # [Batch, Input Length, Channel]   
    x = x.transpose(1, 2)
    x = F.relu(self.node_linear1(x)) 
    x = F.relu(self.node_linear(x))
    x = self.dropout(x)
    x = F.relu(self.node_linear2(x))
    x = x.transpose(1, 2)
    res = x + inputs

    x = self.feature_norm(res)
    x = F.relu(self.feature_linear1(x))  # [Batch, Input Length, FF_Dim]
    x = F.relu(self.feature_linear(x))
    x = self.dropout(x)
    x = F.relu(self.feature_linear2(x))  # [Batch, Input Length, Channel]
    x = self.dropout(x)
    return x + res


class FMixer(nn.Module):
  """FMixer model."""

  def __init__(self, n_block):
    super().__init__()

    # Residual blocks
    self.res_blocks = nn.ModuleList(
        [
            ResBlock()
            for _ in range(n_block)
        ]
    )
    # Output linear
    self.output_linear = nn.Linear(40, 2)

  def forward(self, inputs):
    Batch_size = inputs.size(0)
    # [Batch, Input Length, Channel]
    inputs = inputs.reshape(Batch_size, 2, 20)
    inputs = inputs.reshape(1, 2, 20)
    
    for block in self.res_blocks:
      x = block(inputs)
    x = torch.flatten(x, start_dim=1)
    output = self.output_linear(x)
    output = torch.tanh(output)
    
    return output
  

class EnMixer(nn.Module):
      
  def __init__(self):
    super().__init__()
    device = torch.device("cpu")
    self.model_0 = FMixer(2)
    self.model_1 = FMixer(2)
    self.model_2 = FMixer(2)
    self.model_3 = FMixer(2)
    self.model_4 = FMixer(2)
    # weights_path0 = "setcover_best_model_EnMix_0.ckpt"
    weights_path0 = "liu_setcover_contrast_loss_best_model_EnMix_0.ckpt"
    self.model_0.load_state_dict(torch.load(weights_path0, map_location=device))

    # weights_path1 = "setcover_best_model_EnMix_1.ckpt"
    weights_path1 = "liu_setcover_contrast_loss_best_model_EnMix_1.ckpt"
    self.model_1.load_state_dict(torch.load(weights_path1, map_location=device))

    # weights_path2 = "setcover_best_model_EnMix_2.ckpt"
    weights_path2 = "liu_setcover_contrast_loss_best_model_EnMix_2.ckpt"
    self.model_2.load_state_dict(torch.load(weights_path2, map_location=device))

    # weights_path3 = "setcover_best_model_EnMix_3.ckpt"
    weights_path3 = "liu_setcover_contrast_loss_best_model_EnMix_3.ckpt"
    self.model_3.load_state_dict(torch.load(weights_path3, map_location=device))

    # weights_path4 = "setcover_best_model_EnMix_4.ckpt"
    weights_path4 = "liu_setcover_contrast_loss_best_model_EnMix_4.ckpt"
    self.model_4.load_state_dict(torch.load(weights_path4, map_location=device))

  def forward(self, inputs):
      output0 = self.model_0(inputs)
      output1 = self.model_1(inputs)
      output2 = self.model_2(inputs)
      output3 = self.model_3(inputs)
      output4 = self.model_4(inputs)
      list = [output0, output1, output2, output3, output4]
      concatenated_tensor = torch.cat(list, dim=0)
      mean_tensor = torch.sum(concatenated_tensor, dim=0) / 5
      return mean_tensor
