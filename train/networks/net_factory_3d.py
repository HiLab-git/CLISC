from networks.unet_3D import unet_3D, TriNet, BiNet
# from networks.vnet import VNet
# from networks.VoxResNet import VoxResNet
# from networks.attention_unet import Attention_UNet
# from networks.nnunet import initialize_network


def net_factory_3d(net_type="unet_3D", in_chns=1, class_num=2):
    if net_type == "unet_3D":
        net = unet_3D(n_classes=class_num, in_channels=in_chns).cuda()
    elif net_type == "trinet":
        net = TriNet(n_classes=class_num, in_channels=in_chns).cuda()
    elif net_type == "BiNet":
        net = BiNet(n_classes=class_num, in_channels=in_chns).cuda()
    else:
        net = None
    return net