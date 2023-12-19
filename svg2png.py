# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 19:22:03 2023

@author: skynet
"""
from PIL import Image, ImageDraw, ImageChops

def trimsides(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop((bbox[0],im.height/4 ,bbox[2], 3*im.height/4))

def fulltrim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    else:
        return im
    
def svg2png(file, output):
    with open(file, "r") as f:
        rawdata = f.read()
        data = rawdata.split('<path d="')[1].split(' " ')[0]
        stroke_color = rawdata.split('stroke="')[1].split('" stroke')[0]
        stroke_width = round(float(rawdata.split('stroke-width="')[1].split('" />')[0])*10)
        instructions = data.split()
        
        cursor_pos = [0,0]
        line_pos = []
        canvas = Image.new('RGBA', (10_000, 1_200))
        canvasdraw = ImageDraw.Draw(canvas)   
        
        for instruction in instructions:
            if instruction[:1] == 'M':
                cursor_pos = []
                positions = instruction[1:].split(',')
                for pos in positions:
                    cursor_pos.append(round(float(pos)*10))
            elif instruction[:1] == 'L':
                line_pos = []
                positions = instruction[1:].split(',')
                for pos in positions:
                    line_pos.append(round(float(pos)*10))
                    
                shape = [tuple(cursor_pos), tuple(line_pos)] 
                canvasdraw.line(shape, fill =stroke_color, width = stroke_width) 
                cursor_pos = line_pos
        canvas = canvas.resize((2_000, 240))
        canvas = trimsides(canvas)
        canvas.save(output)
       # canvas.save(output)
        
        #print(instructions)
        
if __name__ == '__main__':
    svg2png('handsynth-temp/0.svg', 'handsynth-temp/1.png')