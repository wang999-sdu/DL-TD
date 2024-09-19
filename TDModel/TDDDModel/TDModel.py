import torch
from torch import nn
from models import resnet


def generate_model(opt):
    assert opt.model in ["resnet"]

    if opt.model == "resnet":
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        if opt.model_depth == 10:
            model = resnet.resnet10(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes,
            )
            # fc_input = 256
            fc_input = 512
        elif opt.model_depth == 18:
            model = resnet.resnet18(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes,
            )
            fc_input = 512
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes,
            )
            fc_input = 512
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes,
            )
            fc_input = 2048
        elif opt.model_depth == 101:
            model = resnet.resnet101(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes,
            )
            fc_input = 2048
        elif opt.model_depth == 152:
            model = resnet.resnet152(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes,
            )
            fc_input = 2048
        elif opt.model_depth == 200:
            model = resnet.resnet200(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes,
            )
            fc_input = 2048
    nb_class = 1
    model.conv_seg = nn.Sequential(
        nn.AdaptiveAvgPool3d((1, 1, 1)),
        nn.Flatten(),
        nn.Linear(in_features=fc_input, out_features=nb_class, bias=True),
    )

    if not opt.no_cuda:
        # if len(opt.gpu_id) > 1:
        #     model = model.cuda()
        #     model = nn.DataParallel(model, device_ids=opt.gpu_id)
        #     net_dict = model.state_dict()
        # else:
        # import os

        # os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id[0])
        print(
            f"os.environ[CUDA_VISIBLE_DEVICES] = {str(opt.gpu_id)} || {torch.cuda.is_available()} || {torch.cuda.device_count()}"
        )
        if torch.cuda.is_available():
            device = torch.device("cuda", opt.gpu_id)
        else:
            device = torch.device("cpu")
        # model = model.cuda()
        model.to(device)
        print(next(model.parameters()).device)
        # model = nn.DataParallel(model, device_ids=None)
        net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()

    # load pretrain
    pretrain = torch.load("./50.pth", map_location="cuda:0")
    pretrain_dict = {
        k: v for k, v in pretrain.items() if k in net_dict.keys()
    }
    net_dict.update(pretrain_dict)
    model.load_state_dict(net_dict)
    # model.load_state_dict(pretrain)
    return model
