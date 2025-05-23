import torch.nn as nn

import MinkowskiEngine as ME
from MinkowskiEngine import MinkowskiNetwork

from policy.minkowski.common import ConvType, NormType, get_norm, conv, sum_pool, avg_pool
from policy.minkowski.resnet_block import BasicBlock, Bottleneck


class ResNetBase(MinkowskiNetwork):
  BLOCK = None
  LAYERS = ()
  INIT_DIM = 64
  PLANES = (64, 128, 256, 512)
  OUT_PIXEL_DIST = 32
  HAS_LAST_BLOCK = False
  CONV_TYPE = ConvType.HYPERCUBE

  def __init__(self, in_channels, out_channels, conv1_kernel_size, strides, dilations, bn_momentum, D=3, init_pool="sum", **kwargs):
    assert self.BLOCK is not None
    assert self.OUT_PIXEL_DIST > 0
    assert init_pool in [None, "sum", "avg"]

    super().__init__(D)

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.init_pool = init_pool
    self.network_initialization(in_channels, out_channels, conv1_kernel_size, strides, dilations, bn_momentum, D)
    self.weight_initialization()

  def network_initialization(self, in_channels, out_channels, conv1_kernel_size, strides, dilations, bn_momentum, D):

    def space_n_time_m(n, m):
      return n if D == 3 else [n, n, n, m]

    if D == 4:
      self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

    self.inplanes = self.INIT_DIM

    if self.init_pool:
      self.conv1 = conv(
          in_channels,
          self.inplanes,
          kernel_size=space_n_time_m(conv1_kernel_size, 1),
          stride=1,
          D=D)

      self.bn1 = get_norm(NormType.BATCH_NORM, self.inplanes, D=self.D, bn_momentum=bn_momentum)
      self.relu = ME.MinkowskiReLU(inplace=True)
      if self.init_pool == "sum":
        self.pool = sum_pool(kernel_size=space_n_time_m(2, 1), stride=space_n_time_m(2, 1), D=D)
      else:
        self.pool = avg_pool(kernel_size=space_n_time_m(2, 1), stride=space_n_time_m(2, 1), D=D)

    self.layer1 = self._make_layer(
        self.BLOCK,
        self.PLANES[0],
        self.LAYERS[0],
        stride=space_n_time_m(strides[0], 1),
        dilation=space_n_time_m(dilations[0], 1))
    self.layer2 = self._make_layer(
        self.BLOCK,
        self.PLANES[1],
        self.LAYERS[1],
        stride=space_n_time_m(strides[1], 1),
        dilation=space_n_time_m(dilations[1], 1))
    self.layer3 = self._make_layer(
        self.BLOCK,
        self.PLANES[2],
        self.LAYERS[2],
        stride=space_n_time_m(strides[2], 1),
        dilation=space_n_time_m(dilations[2], 1))
    self.layer4 = self._make_layer(
        self.BLOCK,
        self.PLANES[3],
        self.LAYERS[3],
        stride=space_n_time_m(strides[3], 1),
        dilation=space_n_time_m(dilations[3], 1))

    self.final = conv(
        self.PLANES[3] * self.BLOCK.expansion, out_channels, kernel_size=1, bias=True, D=D)

  def weight_initialization(self):
    for m in self.modules():
      if isinstance(m, ME.MinkowskiBatchNorm):
        nn.init.constant_(m.bn.weight, 1)
        nn.init.constant_(m.bn.bias, 0)

  def _make_layer(self,
                  block,
                  planes,
                  blocks,
                  stride=1,
                  dilation=1,
                  norm_type=NormType.BATCH_NORM,
                  bn_momentum=0.1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          conv(
              self.inplanes,
              planes * block.expansion,
              kernel_size=1,
              stride=stride,
              bias=False,
              D=self.D),
          get_norm(norm_type, planes * block.expansion, D=self.D, bn_momentum=bn_momentum),
      )
    layers = []
    layers.append(
        block(
            self.inplanes,
            planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            conv_type=self.CONV_TYPE,
            D=self.D))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(
          block(
              self.inplanes,
              planes,
              stride=1,
              dilation=dilation,
              conv_type=self.CONV_TYPE,
              D=self.D))

    return nn.Sequential(*layers)

  def forward(self, x):
    if self.init_pool:
      x = self.conv1(x)
      x = self.bn1(x)
      x = self.relu(x)
      x = self.pool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.final(x)
    return x


class ResNet14Mini(ResNetBase):
  BLOCK = BasicBlock
  LAYERS = (1, 1, 1, 1)
  INIT_DIM = 32
  PLANES = (32, 64, 128, 128)


class ResNet14Max(ResNetBase):
  BLOCK = BasicBlock
  LAYERS = (1, 1, 1, 1)
  INIT_DIM = 256
  PLANES = (256, 256, 512, 512)


class ResNet14(ResNetBase):
  BLOCK = BasicBlock
  LAYERS = (1, 1, 1, 1)


class ResNet18(ResNetBase):
  BLOCK = BasicBlock
  LAYERS = (2, 2, 2, 2)


class ResNet34(ResNetBase):
  BLOCK = BasicBlock
  LAYERS = (3, 4, 6, 3)


class ResNet50(ResNetBase):
  BLOCK = Bottleneck
  LAYERS = (3, 4, 6, 3)


class ResNet101(ResNetBase):
  BLOCK = Bottleneck
  LAYERS = (3, 4, 23, 3)