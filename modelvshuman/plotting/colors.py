"""
Define color scheme.
"""


def rgb(r, g, b, divide_by=255.0):
    """Convenience function: return colour in [0, 1]."""
    return (r/divide_by, g/divide_by, b/divide_by)


# primary colors
red = rgb(165, 30, 55)
gold = rgb(180, 160, 105)
metallic = rgb(50, 65, 75)

# secondary colors
blue1 = rgb(65, 90, 140)
blue2 = rgb(0, 105, 170)
blue3 = rgb(80, 170, 200)

green1 = rgb(50, 110, 30)
green2 = rgb(125, 165, 75)
green3 = rgb(130, 185, 160)

grey1 = rgb(180, 160, 150)
purple1 = rgb(175, 110, 150)
red1 = rgb(200, 80, 60)

brown1 = rgb(145, 105, 70)
orange1 = rgb(210, 150, 0)
orange2 = rgb(215, 180, 105)

# other colors
black = rgb(0, 0, 0)
