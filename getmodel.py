import models


def GetModel(args):
    backbone_model = models.__dict__[args.arch]()
    if args.net == 'Raw':
        model = backbone_model
    elif args.net == 'Seg':
        model = models.Segmodel(backbone_model,
                                fchannel=args.fchannel,
                                classes=args.classes,
                                dropout=args.dropout,
                                zoom_factor=args.zoom_factor)
    elif args.net == 'ReSeg':
        model = models.ReSegmodel(backbone_model,
                                  fchannel=args.fchannel,
                                  hiddenchannel=args.hiddenchannel,
                                  renet_num=args.renet_num,
                                  classes=args.classes,
                                  dropout=args.dropout,
                                  zoom_factor=args.zoom_factor)
    elif args.net == 'ReSegRes':
        model = models.ReSegResmodel(backbone_model,
                                     fchannel=args.fchannel,
                                     hiddenchannel=args.hiddenchannel,
                                     renet_num=args.renet_num,
                                     classes=args.classes,
                                     dropout=args.dropout,
                                     zoom_factor=args.zoom_factor)
    else:
        raise ValueError("args.net type error.\n")
    return model
