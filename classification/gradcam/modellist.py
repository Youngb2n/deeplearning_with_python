from model import vgg, resnet, densenet, efficientnet, mobilenetv3


def Modellist(model_name, num_classes = 1000, use_attention=None):
    if model_name == 'vgg11':
        return vgg.VGG11(num_classes, use_attention=use_attention)
    elif model_name == 'vgg13':
        return vgg.VGG13(num_classes, use_attention=use_attention)
    elif model_name == 'vgg16':
        return vgg.VGG16(num_classes, use_attention=use_attention)
    elif model_name == 'vgg19':
        return vgg.VGG19(num_classes, use_attention=use_attention)
    elif model_name == 'resnet18':
        return resnet.ResNet18(num_classes, use_attention=use_attention)
    elif model_name == 'resnet34':
        return resnet.ResNet34(num_classes, use_attention=use_attention)
    elif model_name == 'resnet50':
        return resnet.ResNet50(num_classes, use_attention=use_attention)
    elif model_name == 'resnet101':
        return resnet.ResNet101(num_classes, use_attention=use_attention)
    elif model_name == 'resnet152':
        return resnet.ResNet152(num_classes, use_attention=use_attention)
    elif model_name == 'densenet121':
        return densenet.DenseNet121(num_classes, use_attention=use_attention)
    elif model_name == 'densenet169':
        return densenet.DenseNet169(num_classes, use_attention=use_attention)
    elif model_name == 'densenet201':
        return densenet.DenseNet201(num_classes, use_attention=use_attention)
    elif model_name == 'densenet161':
        return densenet.DenseNet161(num_classes, use_attention=use_attention)
    elif model_name == 'mobilenetv3_small':
        return mobilenetv3.mobilenetv3_small(num_classes)
    elif model_name == 'mobilenetv3_large':
        return mobilenetv3.mobilenetv3_large(num_classes)
    elif model_name == 'efficientnet_b0':
        return efficientnet.efficientnet_b0(num_classes, use_attention=use_attention)    
    elif model_name == 'efficientnet_b1':
        return efficientnet.efficientnet_b1(num_classes, use_attention=use_attention)    
    elif model_name == 'efficientnet_b2':
        return efficientnet.efficientnet_b2(num_classes, use_attention=use_attention)    
    elif model_name == 'efficientnet_b3':
        return efficientnet.efficientnet_b3(num_classes, use_attention=use_attention)
    elif model_name == 'efficientnet_b4':
        return efficientnet.efficientnet_b4(num_classes, use_attention=use_attention)
    elif model_name == 'efficientnet_b5':
        return efficientnet.efficientnet_b5(num_classes, use_attention=use_attention)
    elif model_name == 'efficientnet_b6':
        return efficientnet.efficientnet_b6(num_classes, use_attention=use_attention)
    else:
        raise ValueError("The model_name does not exist.")
        
