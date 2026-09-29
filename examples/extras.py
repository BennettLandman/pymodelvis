"""Interactive HTML explorer, the stage-by-stage "light-up" animation, and layout/theme variants."""
import torch

from _common import out_path
from real_models import imagenet_classes, load_resnet50, photo, to_input
from neural_flow import animate_model, visualize_model


def main():
    model, mean, std = load_resnet50()
    x = to_input(photo("chelsea"), mean, std)
    names = imagenet_classes()
    visualize_model(model, x, output=out_path("resnet50_interactive.html"), class_names=names, top_k=5)
    visualize_model(model, x, output=out_path("resnet50_cinematic_169.png"), style="cinematic", class_names=names,
                    top_k=5, figsize=(16, 9), dpi=150, title="ResNet-50")
    visualize_model(model, x, output=out_path("resnet50_cinematic_light.png"), style="cinematic", theme="light",
                    class_names=names, top_k=5, title="ResNet-50")
    visualize_model(model, x, output=out_path("resnet50_cinematic.svg"), style="cinematic", class_names=names,
                    top_k=5, title="ResNet-50")
    animate_model(model, x, output=out_path("resnet50_lightup.gif"), class_names=names, dpi=70)
    print("wrote HTML / 16:9 / light / SVG / light-up GIF to", out_path(""))


if __name__ == "__main__":
    main()
