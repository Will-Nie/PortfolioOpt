import numpy as np
import torch
import torch.nn as nn
from .nn_module import fc_block, MLP
from .old_transformer import Transformer as OldTransformer
from .transformer import Transformer


class OnehotEncoder(nn.Module):

    def __init__(self, num_embeddings: int):
        super(OnehotEncoder, self).__init__()
        self.num_embeddings = num_embeddings
        self.main = nn.Embedding.from_pretrained(torch.eye(self.num_embeddings), freeze=True, padding_idx=None)

    def forward(self, x: torch.Tensor):
        x = x.long()
        return self.main(x)


class OnehotEmbedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super(OnehotEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.main = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim)

    def forward(self, x: torch.Tensor):
        x = x.long()
        return self.main(x)


class BinaryEncoder(nn.Module):

    def __init__(self, num_embeddings: int):
        super(BinaryEncoder, self).__init__()
        self.bit_num = num_embeddings
        self.main = nn.Embedding.from_pretrained(
            self.get_binary_embed_matrix(self.bit_num), freeze=True, padding_idx=None
        )

    @staticmethod
    def get_binary_embed_matrix(bit_num):
        embedding_matrix = []
        for n in range(2 ** bit_num):
            embedding = [n >> d & 1 for d in range(bit_num)][::-1]
            embedding_matrix.append(embedding)
        return torch.tensor(embedding_matrix, dtype=torch.float)

    def forward(self, x: torch.Tensor):
        x = x.long()
        x.clamp_(max=2 ** self.bit_num - 1)
        return self.main(x)


class SignBinaryEncoder(nn.Module):

    def __init__(self, num_embeddings):
        super(SignBinaryEncoder, self).__init__()
        self.bit_num = num_embeddings
        self.main = nn.Embedding.from_pretrained(
            self.get_sign_binary_matrix(self.bit_num), freeze=True, padding_idx=None
        )
        self.max_val = 2 ** (self.bit_num - 1) - 1

    @staticmethod
    def get_sign_binary_matrix(bit_num):
        neg_embedding_matrix = []
        pos_embedding_matrix = []
        for n in range(1, 2 ** (bit_num - 1)):
            embedding = [n >> d & 1 for d in range(bit_num - 1)][::-1]
            neg_embedding_matrix.append([1] + embedding)
            pos_embedding_matrix.append([0] + embedding)
        embedding_matrix = neg_embedding_matrix[::-1] + [[0 for _ in range(bit_num)]] + pos_embedding_matrix
        return torch.tensor(embedding_matrix, dtype=torch.float)

    def forward(self, x: torch.Tensor):
        x = x.long()
        x.clamp_(max=self.max_val, min=-self.max_val)
        return self.main(x + self.max_val)


class PositionEncoder(nn.Module):

    def __init__(self, num_embeddings, embedding_dim=None):
        super(PositionEncoder, self).__init__()
        self.n_position = num_embeddings
        self.embedding_dim = self.n_position if embedding_dim is None else embedding_dim
        self.position_enc = nn.Embedding.from_pretrained(
            self.position_encoding_init(self.n_position, self.embedding_dim), freeze=True, padding_idx=None
        )

    @staticmethod
    def position_encoding_init(n_position, embedding_dim):
        ''' Init the sinusoid position encoding table '''

        # keep dim 0 for padding token position encoding zero vector
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / embedding_dim) for j in range(embedding_dim)]
                for pos in range(n_position)
            ]
        )
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # apply sin on 0th,2nd,4th...embedding_dim
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # apply cos on 1st,3rd,5th...embedding_dim
        return torch.from_numpy(position_enc).type(torch.FloatTensor)

    def forward(self, x: torch.Tensor):
        return self.position_enc(x)


class TimeEncoder(nn.Module):

    def __init__(self, embedding_dim):
        super(TimeEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.position_array = torch.nn.Parameter(self.get_position_array(), requires_grad=False)

    def get_position_array(self):
        x = torch.arange(0, self.embedding_dim, dtype=torch.float)
        x = x // 2 * 2
        x = torch.div(x, self.embedding_dim)
        x = torch.pow(10000., x)
        x = torch.div(1., x)
        return x

    def forward(self, x: torch.Tensor):
        v = torch.zeros(size=(x.shape[0], self.embedding_dim), dtype=torch.float, device=x.device)
        assert len(x.shape) == 1
        x = x.unsqueeze(dim=1)
        v[:, 0::2] = torch.sin(x * self.position_array[0::2])  # even
        v[:, 1::2] = torch.cos(x * self.position_array[1::2])  # odd
        return v


class UnsqueezeEncoder(nn.Module):

    def __init__(self, unsqueeze_dim: int = -1, norm_value: float = 1):
        super(UnsqueezeEncoder, self).__init__()
        self.unsqueeze_dim = unsqueeze_dim
        self.norm_value = norm_value

    def forward(self, x: torch.Tensor):
        x = x.float().unsqueeze(dim=self.unsqueeze_dim)
        if self.norm_value != 1:
            x = x / self.norm_value
        return x


class BeginningBuildOrderEncoder(nn.Module):

    def __init__(self, cfg, bo_cfg):
        super(BeginningBuildOrderEncoder, self).__init__()
        self.whole_cfg = cfg
        self.spatial_size = self.whole_cfg.agent.features.spatial_size
        self.beginning_order_length = self.whole_cfg.agent.features.beginning_order_length  # 20
        self.cfg = bo_cfg
        self.input_dim = self.cfg.action_one_hot_dim + self.beginning_order_length + self.cfg.binary_dim * 2

        self.output_dim = self.cfg.output_dim
        self.embedding_dim = self.output_dim
        self.norm_type = self.cfg.norm_type
        self.activation_type = self.cfg.activation

        self.encode_layers = MLP(
            in_channels=self.input_dim,
            hidden_channels=self.embedding_dim,
            out_channels=self.embedding_dim,
            layer_num=1,
            layer_fn=fc_block,
            activation=self.activation_type,
            norm_type=self.norm_type,
            use_dropout=False
        )

        self.transformer_cfg = self.cfg.transformer
        self.transformer = Transformer(
            n_heads=self.transformer_cfg.head_num,
            embedding_size=self.embedding_dim,
            ffn_size=self.transformer_cfg.ffn_size,
            n_layers=self.transformer_cfg.layer_num,
            attention_dropout=0.0,
            relu_dropout=0.0,
            dropout=0.0,
            activation=self.activation_type,
            variant=self.transformer_cfg.variant,
        )

        self.embedd_fc = fc_block(self.embedding_dim, self.output_dim, activation=self.activation_type)
        self.action_one_hot = OnehotEncoder(num_embeddings=self.cfg.action_one_hot_dim)
        self.order_one_hot = OnehotEncoder(num_embeddings=self.beginning_order_length)
        self.location_binary = BinaryEncoder(num_embeddings=self.cfg.binary_dim)

    def _add_seq_info(self, x):
        indices_one_hot = torch.zeros(size=(x.shape[1], x.shape[1]), device=x.device)
        indices = torch.arange(x.shape[1], device=x.device).unsqueeze(dim=1)
        indices_one_hot = indices_one_hot.scatter_(dim=-1, index=indices, value=1.)
        indices_one_hot = indices_one_hot.unsqueeze(0).repeat(x.shape[0], 1, 1)  # expand to batch dim
        return torch.cat([x, indices_one_hot], dim=2)

    def forward(self, x, bo_location):
        x = x.float()
        bo_location = bo_location.long()
        x = self.action_one_hot(x)
        x = self._add_seq_info(x)
        location_x = bo_location % self.spatial_size[1]
        location_y = bo_location // self.spatial_size[1]
        location_x = self.location_binary(location_x)
        location_y = self.location_binary(location_y)
        x = torch.cat([x, location_x, location_y], dim=2)
        assert len(x.shape) == 3
        x = self.encode_layers(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.embedd_fc(x)
        return x


class OldBeginningBuildOrderEncoder(nn.Module):

    def __init__(self, cfg, bo_cfg):
        super(OldBeginningBuildOrderEncoder, self).__init__()
        self.whole_cfg = cfg
        self.spatial_size = self.whole_cfg.agent.features.spatial_size
        self.beginning_order_length = self.whole_cfg.agent.features.beginning_order_length  # 20
        self.cfg = bo_cfg
        self.output_dim = self.cfg.output_dim
        self.input_dim = self.cfg.action_one_hot_dim + self.beginning_order_length + self.cfg.binary_dim * 2
        self.activation_type = self.cfg.activation
        self.transformer = OldTransformer(
            input_dim=self.input_dim,
            head_dim=self.cfg.head_dim,
            hidden_dim=self.cfg.output_dim * 2,
            output_dim=self.cfg.output_dim
        )
        self.embedd_fc = fc_block(self.cfg.output_dim, self.output_dim, activation=self.activation_type)
        self.action_one_hot = OnehotEncoder(num_embeddings=self.cfg.action_one_hot_dim)
        self.order_one_hot = OnehotEncoder(num_embeddings=self.beginning_order_length)
        self.location_binary = BinaryEncoder(num_embeddings=self.cfg.binary_dim)

    def _add_seq_info(self, x):
        indices_one_hot = torch.zeros(size=(x.shape[1], x.shape[1]), device=x.device)
        indices = torch.arange(x.shape[1], device=x.device).unsqueeze(dim=1)
        indices_one_hot = indices_one_hot.scatter_(dim=-1, index=indices, value=1.)
        indices_one_hot = indices_one_hot.unsqueeze(0).repeat(x.shape[0], 1, 1)  # expand to batch dim
        return torch.cat([x, indices_one_hot], dim=2)

    def forward(self, x, bo_location):
        x = x.float()
        bo_location = bo_location.long()
        x = self.action_one_hot(x)
        x = self._add_seq_info(x)
        location_x = bo_location % self.spatial_size[1]
        location_y = bo_location // self.spatial_size[1]
        location_x = self.location_binary(location_x)
        location_y = self.location_binary(location_y)
        x = torch.cat([x, location_x, location_y], dim=2)
        assert len(x.shape) == 3
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.embedd_fc(x)
        return x


if __name__ == '__main__':
    pass
