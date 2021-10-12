# /!usr/bin/env python3

"""
Define decision makers (either human participants or CNN models).
"""

from modelvshuman import constants as c
from modelvshuman.plotting.colors import *
from modelvshuman.plotting.decision_makers import DecisionMaker


def plotting_definition_template(df):
    """Decision makers to compare a few models with human observers.

    This exemplary definition can be adapted for the
    desired purpose, e.g. by adding more/different models.

    Note that models will need to be evaluated first, before
    their data can be plotted.

    For each model, define:
    - a color using rgb(42, 42, 42)
    - a plotting symbol by setting marker;
      a list of markers can be found here:
      https://matplotlib.org/3.1.0/api/markers_api.html
    """

    decision_makers = []

    decision_makers.append(DecisionMaker(name_pattern="resnet50",
                           color=rgb(65, 90, 140), marker="o", df=df,
                           plotting_name="ResNet-50"))
    decision_makers.append(DecisionMaker(name_pattern="bagnet33",
                           color=rgb(110, 110, 110), marker="o", df=df,
                           plotting_name="BagNet-33"))
    decision_makers.append(DecisionMaker(name_pattern="simclr_resnet50x1",
                           color=rgb(210, 150, 0), marker="o", df=df,
                           plotting_name="SimCLR-x1"))
    decision_makers.append(DecisionMaker(name_pattern="subject-*",
                           color=rgb(165, 30, 55), marker="D", df=df,
                           plotting_name="humans"))
    return decision_makers


def get_comparison_decision_makers(df, include_humans=True,
                                   humans_last=True):
    """Decision makers used in our paper."""

    d = []

    # 1. supervised models
    for model in c.TORCHVISION_MODELS:
        d.append(DecisionMaker(name_pattern=model,
                               color=rgb(230, 230, 230), df=df,
                               plotting_name=model))

    # 2. self-supervised models
    for model in c.PYCONTRAST_MODELS:
        d.append(DecisionMaker(name_pattern=model,
                               color=orange2, marker="o", df=df,
                               plotting_name=model+": ResNet-50"))
    d.append(DecisionMaker(name_pattern="simclr_resnet50x1",
                           color=orange2, marker="o", df=df,
                           plotting_name="SimCLR: ResNet-50x1"))
    d.append(DecisionMaker(name_pattern="simclr_resnet50x2",
                           color=orange2, marker="o", df=df,
                           plotting_name="SimCLR: ResNet-50x2"))
    d.append(DecisionMaker(name_pattern="simclr_resnet50x4",
                           color=orange2, marker="o", df=df,
                           plotting_name="SimCLR: ResNet-50x4"))


    # 3. adversarially robust models
    d += [DecisionMaker(name_pattern="resnet50_l2_eps0",
                        color=rgb(196, 205, 229), marker="o", df=df,
                        plotting_name="ResNet-50 L2 eps 0.0"),
          DecisionMaker(name_pattern="resnet50_l2_eps0_5",
                        color=rgb(176, 190, 220), marker="o", df=df,
                        plotting_name="ResNet-50 L2 eps 0.5"),
          DecisionMaker(name_pattern="resnet50_l2_eps1",
                        color=rgb(134, 159, 203), marker="o", df=df,
                        plotting_name="ResNet-50 L2 eps 1.0"),
          DecisionMaker(name_pattern="resnet50_l2_eps3",
                        color=rgb(86, 130, 186), marker="o", df=df,
                        plotting_name="ResNet-50 L2 eps 3.0"),
          DecisionMaker(name_pattern="resnet50_l2_eps5",
                        color=blue2, marker="o", df=df,
                        plotting_name="ResNet-50 L2 eps 5.0")]

    # 4. vision transformers without large-scale pretraining
    d.append(DecisionMaker(name_pattern="vit_small_patch16_224",
                           color=rgb(144, 159, 110), marker="v", df=df,
                           plotting_name="ViT-S"))
    d.append(DecisionMaker(name_pattern="vit_base_patch16_224",
                           color=rgb(144, 159, 110), marker="v", df=df,
                           plotting_name="ViT-B"))
    d.append(DecisionMaker(name_pattern="vit_large_patch16_224",
                           color=rgb(144, 159, 110), marker="v", df=df,
                           plotting_name="ViT-L"))

    if not humans_last:
        if include_humans:
            d.append(DecisionMaker(name_pattern="subject-*",
                                   color=red, marker="D", df=df,
                                   plotting_name="humans"))
        d.append(DecisionMaker(name_pattern="clip",
                               color=brown1, marker="v", df=df,
                               plotting_name="CLIP: ViT-B (400M)"))
 
    d.append(DecisionMaker(name_pattern="ResNeXt101_32x16d_swsl",
                           color=purple1, marker="o", df=df,
                           plotting_name="SWSL: ResNeXt-101 (940M)"))
    d.append(DecisionMaker(name_pattern="resnet50_swsl",
                           color=purple1, marker="o", df=df,
                           plotting_name="SWSL: ResNet-50 (940M)"))
 
    bitm_col = rgb(153, 142, 195) 
    d.append(DecisionMaker(name_pattern="BiTM_resnetv2_152x4",
                           color=bitm_col, marker="o", df=df,
                           plotting_name="BiT-M: ResNet-152x4 (14M)"))
    d.append(DecisionMaker(name_pattern="BiTM_resnetv2_152x2",
                           color=bitm_col, marker="o", df=df,
                           plotting_name="BiT-M: ResNet-152x2 (14M)"))
    d.append(DecisionMaker(name_pattern="BiTM_resnetv2_101x3",
                           color=bitm_col, marker="o", df=df,
                           plotting_name="BiT-M: ResNet-101x3 (14M)"))
    d.append(DecisionMaker(name_pattern="BiTM_resnetv2_101x1",
                           color=bitm_col, marker="o", df=df,
                           plotting_name="BiT-M: ResNet-101x1 (14M)"))
    d.append(DecisionMaker(name_pattern="BiTM_resnetv2_50x3",
                           color=bitm_col, marker="o", df=df,
                           plotting_name="BiT-M: ResNet-50x3 (14M)"))
    d.append(DecisionMaker(name_pattern="BiTM_resnetv2_50x1",
                           color=bitm_col, marker="o", df=df,
                           plotting_name="BiT-M: ResNet-50x1 (14M)"))

    d.append(DecisionMaker(name_pattern="transformer_L16_IN21K",
                           color=green1, marker="v", df=df,
                           plotting_name="ViT-L (14M)"))
    d.append(DecisionMaker(name_pattern="transformer_B16_IN21K",
                           color=green1, marker="v", df=df,
                           plotting_name="ViT-B (14M)"))

    d.append(DecisionMaker(name_pattern="efficientnet_l2_noisy_student_475",
                           color=metallic, marker="o", df=df,
                           plotting_name="Noisy Student: ENetL2 (300M)"))
 
    if humans_last:
        d.append(DecisionMaker(name_pattern="clip",
                               color=brown1, marker="v", df=df,
                               plotting_name="CLIP: ViT-B (400M)"))
        if include_humans:
            d.append(DecisionMaker(name_pattern="subject-*",
                                   color=red, marker="D", df=df,
                                   plotting_name="humans"))

    return d
