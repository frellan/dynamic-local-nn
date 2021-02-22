import torch

def get_unsupervised_weights(X, n_hidden, n_epochs, batch_size,
        learning_rate=2e-2, precision=1e-30, anti_hebbian_learning_strength=0.4, lebesgue_norm=2.0, rank=2):
    sample_sz = X.shape[1]
    weights = torch.rand((n_hidden, sample_sz), dtype=torch.float).cuda()
    for epoch in range(n_epochs):
        eps = learning_rate * (1 - epoch / n_epochs)
        shuffled_epoch_data = X[torch.randperm(X.shape[0]),:]
        for i in range(X.shape[0] // batch_size):
            mini_batch = shuffled_epoch_data[i*batch_size:(i+1)*batch_size,:].cuda()
            mini_batch = torch.transpose(mini_batch, 0, 1)
            sign = torch.sign(weights)
            W = sign * torch.abs(weights) ** (lebesgue_norm - 1)
            tot_input=torch.mm(W, mini_batch)

            y = torch.argsort(tot_input, dim=0)
            yl = torch.zeros((n_hidden, batch_size), dtype = torch.float).cuda()
            yl[y[n_hidden-1,:], torch.arange(batch_size)] = 1.0
            yl[y[n_hidden-rank], torch.arange(batch_size)] =- anti_hebbian_learning_strength

            xx = torch.sum(yl * tot_input,1)
            xx = xx.unsqueeze(1)
            xx = xx.repeat(1, sample_sz)
            ds = torch.mm(yl, torch.transpose(mini_batch, 0, 1)) - xx * weights

            nc = torch.max(torch.abs(ds))
            if nc < precision: nc = precision
            weights += eps*(ds/nc)
    return weights
