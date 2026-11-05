from captum.attr import LayerAttribution, LayerGradCam


def grad_cam(model, specs, target=1, layer="net.layer4"):
    model.eval()
    attributions = LayerGradCam(model, model.get_submodule(layer)).attribute(
        specs, target=target, relu_attributions=True
    )
    return LayerAttribution.interpolate(attributions, specs.shape[-2:], interpolate_mode="bilinear").detach()
