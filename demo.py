import numpy as np

import lyrics
from hand import Hand

if __name__ == '__main__':
    hand = Hand()

    # usage demo
    lines = [
        "Now this is a story all about how",
        "My life got flipped turned upside down",
        "And I'd like to take a minute, just sit right there",
        "I'll tell you how I became the prince of a town called Bel-Air",
    ]
    biases = [.75 for i in lines]
    styles = [9 for i in lines]
    stroke_colors = ['red', 'green', 'black', 'blue']
    stroke_widths = [1, 2, 1, 2]

    hand.write(
        filename='img/usage_demo.svg',
        lines=lines,
        biases=biases,
        styles=styles,
        stroke_colors=stroke_colors,
        stroke_widths=stroke_widths
    )

    # demo number 1 - fixed bias, fixed style
    lines = lyrics.all_star.split("\n")
    biases = [.75 for i in lines]
    styles = [12 for i in lines]

    hand.write(
        filename='img/all_star.svg',
        lines=lines,
        biases=biases,
        styles=styles,
    )

    # demo number 2 - fixed bias, varying style
    lines = lyrics.downtown.split("\n")
    biases = [.75 for i in lines]
    styles = np.cumsum(np.array([len(i) for i in lines]) == 0).astype(int)

    hand.write(
        filename='img/downtown.svg',
        lines=lines,
        biases=biases,
        styles=styles,
    )

    # demo number 3 - varying bias, fixed style
    lines = lyrics.give_up.split("\n")
    biases = .2*np.flip(np.cumsum([len(i) == 0 for i in lines]), 0)
    styles = [7 for i in lines]

    hand.write(
        filename='img/give_up.svg',
        lines=lines,
        biases=biases,
        styles=styles,
    )
