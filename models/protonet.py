import os
import sys
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter

sys.path.append(os.path.dirname(os.getcwd()))
import torch.nn as nn
from utils import euclidean_metric
from networks.convnet import ConvNet


class ProtoNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        model_name = args.model

        if model_name == "convnet":
            self.encoder = ConvNet(z_dim=args.dim)
        else:
            raise ValueError("Model not found")


    def forward(self, data_shot, data_query):
        proto = self.encoder(data_shot)

        if self.training:
            proto = proto.reshape(self.args.shot, self.args.way, -1).mean(dim=0)
        else:
            proto = proto.reshape(
                self.args.shot, self.args.validation_way, -1
            ).mean(dim=0)

        logits = (
            euclidean_metric(self.encoder(data_query), proto)
            / self.args.temperature
        )
        return logits


    def val_encode(self,data_shot):
        proto = self.encoder(data_shot)
        if self.args.norm:
            proto=F.normalize(proto)
        elif self.args.hyperbolic:
            proto = self.e2p(proto)
        return proto

    def encode(self,data_shot):
        proto = self.encoder(data_shot)
        if self.args.norm:
            proto=F.normalize(proto)
        elif self.args.hyperbolic:
            proto = self.e2p(proto)
        return proto

    def val_decode(self,proto,data_query):
        logits = (
                euclidean_metric(self.encoder(data_query), proto)
                / self.args.temperature
        )
        return logits

    def decode(self,proto,data_query):
        logits = (
                euclidean_metric(data_query, proto)
                / self.args.temperature
        )
        return logits