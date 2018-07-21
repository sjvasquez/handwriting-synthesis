from demo import Hand
import random
import argparse

class HandwriteCLI():

    def __init__(self):
        self.args = self.get_parsed_args()

    '''Handwrite the supplied text and print a report if requested.'''
    def run(self):
        lines = self.get_input_lines()
        self.handwrite(lines)
        if not self.args.silent:
            self.print_report()

    def get_parsed_args(self):
        # Default values
        outfile = 'output.svg'
        color = 'black'
        background_color = 'none'
        bias = 0.5
        style = 0
        width = 1

        # Parse commandline arguments
        parser = argparse.ArgumentParser(prog='handwriting-synthesis',
                description='Write handwritten note to svg.')

        parser.add_argument('--outfile', nargs='?', default=outfile, \
            help='Choose destination file. Defaults to {}.'.format(outfile))

        text_warning = '''
        Max line length 75 chars.
        Valid character set is a-z, A-Z, 0-9, and '\x00', '!',
         ' ', '#', '"', "'", ')', '(', '-', ',', '.'
        '''
        input_group = parser.add_mutually_exclusive_group(required=True)
        input_group.add_argument('--infile', nargs='?', default=None, \
            help='Choose file containing text to handwrite.' + text_warning)
        input_group.add_argument('--text', nargs='?', default=None, \
            help='Supply text to handwrite.' + text_warning)

        color_warning = 'Available colors are SVG color names, \
             "none" for transparent, or hex codes in the format "#000000".'
        parser.add_argument('--color', nargs='?', default=color, \
            help='Choose color. Defaults to {}. {}'.format(color, color_warning))
        parser.add_argument('--bgcolor', nargs='?', default=background_color, \
            help='Choose background color. Defaults to {}. {}'.format(background_color, color_warning))

        parser.add_argument('--bias', type=float, nargs='?', default=bias, \
            help='Choose bias between 0 and 1. Defaults to {}.'.format(bias))

        parser.add_argument('--style', type=int, nargs='?', default=style, \
            help='Choose style between 0 and 12 inclusive. View styles in /style/. Defaults to {}.'.format(style))

        parser.add_argument('--width', type=int, nargs='?', default=width, \
            help='Choose stroke width between 1 and 5 inclusive. Defaults to {}.'.format(width))

        parser.add_argument('--silent', default=False, action='store_true', \
            help='Suppress output to terminal. Defaults to false.')

        args = parser.parse_args()
        return args

    def get_input_lines(self, ):
        if self.args.infile is not None:
            lines = open(self.args.infile, 'r').readlines()
            lines = list(map(str.rstrip, lines))
        else:
            lines = self.args.text.splitlines()
        return lines

    def handwrite(self, lines):

        length = len(lines)

        # Use the same style for each line
        biases = [self.args.bias] * length
        styles = [self.args.style] * length
        stroke_colors = [self.args.color] * length
        stroke_widths = [self.args.width] * length

        # Write note
        hand = Hand()
        hand.write(
            filename=self.args.outfile,
            lines=lines,
            biases=biases,
            styles=styles,
            stroke_colors=stroke_colors,
            stroke_widths=stroke_widths,
            background_color=self.args.bgcolor
        )

    def print_report(self):
        if self.args.infile is not None:
            input_str = 'Contents of ' + self.args.infile
        else:
            input_str = 'Supplied text'
        print('''{input_str} written to {self.args.outfile}.
Color {self.args.color}, background color {self.args.bgcolor},
style {self.args.style}, bias {self.args.bias}, width {self.args.width}.
-h for more information.'''.format(**locals()))

if __name__ == '__main__':
    cli = HandwriteCLI()
    cli.run()
