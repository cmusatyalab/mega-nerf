import torch
from torch import nn

from mega_nerf.models.nerf import Embedding


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
    """
    SDF Network
    """
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        """ Initialization

        Args:
            d_in (int): input dimension of SDF network, set 257 for most cases
            d_out (int): output dimension of SDF network, set 3 for most cases
            d_hidden (int): number of neurons per hidden layer, set 256 for most cases
            n_layers (_type_): number of layers in total, set 8 for most cases
            skip_in (tuple, optional): skip connection layer (extra embedded input). Defaults to (4,).
            multires (int, optional): resolution dim for embedded input. Defaults to 0.
            bias (float, optional): initialized biases. Defaults to 0.5.
            scale (int, optional): . Defaults to 1.
            geometric_init (bool, optional): initialize weights and biases. Defaults to True.
            weight_norm (bool, optional): . Defaults to True.
            inside_outside (bool, optional): . Defaults to False.
        """
        super(SDFNetwork, self).__init__()

        """
        for input layer, the dimension of input is d_in
        for hidden layer, the dimension of input is d_hidden
            - the skip connection not considered yet.
        for output layer, the dimension of input is d_out
        """
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            """
            if multires > 0, it means that we want to use the multiresolution embedding
            will change input_channel from d_in to the embedder out_dim
            """
            # embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            embed_fn = Embedding(multires)
            input_ch = embed_fn.get_out_channels(d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                """
                if the next layer is in skip layers, then the output of current layer should
                be subtracted from the input layer number to satisfy the input dimension 
                requirement of the next layer
                """
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                """
                initialize weights and biases
                """
                if l == self.num_layers - 2:
                    """
                    the last hidden layer
                    """
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        """ Forward pass of SDF network

        Args:
            inputs (Tensor): input tensor of shape (batch_size, d_in)

        Returns:
            Tensor: output tensor of shape (batch_size, d_out)
        """
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        """ only keep the first dim of the forward pass

        Args:
            x (Tensor): input tensor of shape (batch_size, d_in)

        Returns:
            output tensor of shape (batch_size, 1)
        """
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)