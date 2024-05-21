from torch.optim import Adam

def MakeAdam(model, lr=0.001):
    return Adam(model.parameters(), lr=lr)