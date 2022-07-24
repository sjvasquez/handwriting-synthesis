from hand import Hand


if __name__ == '__main__':

    hand = Hand()

    lines = [
        "Writer has written",
        "One writes in kind",
        "You see the hand",
        "You see the mind",
        "Source begets source",
        "Begets source",
        "Begets source"
    ]
    
    biases = [.75 for i in lines]
    styles = [5 for i in lines]
    stroke_colors = ['black' for i in lines]
    stroke_widths = [1 for i in lines]
    
    hand.write(
        filename='img/source.svg',
        lines=lines,
        biases=biases,
        styles=styles,
        stroke_colors=stroke_colors,
        stroke_widths=stroke_widths,
        center_align=False,
        output_png=True
    )
