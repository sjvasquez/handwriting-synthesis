# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 20:52:50 2023

@author: skynet
"""
from demo import Hand
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from tkcolorpicker import askcolor
from PIL import ImageTk, Image
import subprocess
import os
from threading import Thread
import time
import math
from svg2png import svg2png, fulltrim
import shutil
import string
import random
from resourcepath import resourcepath

class MyWindow:
    def __init__(self, win):
        self.text = ''
        self.stylelabel = None
        self.colorlabel = None
        self.colortextlabel = None
        self.hand = None
        self.styles = []
        self.currentstyle = 0
        self.stroke_color = '#003264'      
        self.valid_chars = ['"', '6', 'G', 'b', ')', 'I', '0', 'O', 'c', '9', 'L', '8', 't', 'q', 's', '\x00', 'U', 'S', 'W', 'a', 'k', '2', 'B', 'M', '7', 'T', 'g', 'f', 'F', 'P', 'l', 'E', 'v', 'y', 'j', 'Y', 'J', '-', 'R', '!', '#', '.', 'o', 'r', '?', 'C', "'", '5', 'm', 'h', '4', 'A', 'u', 'p', 'w', 'n', '(', 'V', 'd', '1', ',', 'H', 'i', 'x', ';', ':', 'z', 'K', '3', 'N', ' ', 'D', 'e', '\n']
    
        for i in range(12):
            styleimage = Image.open(resourcepath(f"gui/stylebook/images/{i}.png")).crop((400, 30, 620, 90))
            self.styles.append(styleimage)
            
        self.legibilityscale = ttk.Scale(win, length=200, from_=0, to=100, orient=tk.HORIZONTAL, value=50) #tickinterval=20)
        self.legibilityscale.place(x=32, y=75)
        
        ttk.Label(win, text='Legibility',font=('Helvetica bold',16)).place(x=90, y=20)
        ttk.Label(win, text='Illegible').place(x=16, y=31)
        ttk.Label(win, text='Legible').place(x=208, y=31)
        
        
        self.widthscale = ttk.Scale(win, length=200, from_=0, to=10, orient=tk.HORIZONTAL, value=5)# tickinterval=1)
        self.widthscale.place(x=32, y=185)
        
        ttk.Label(win, text='Stroke Width',font=('Helvetica bold',16)).place(x=72, y=130)
        ttk.Label(win, text='Thin').place(x=22, y=141)
        ttk.Label(win, text='Thick').place(x=212, y=141)
        
        ttk.Label(win, text='Writing Style',font=('Helvetica bold',16)).place(x=75, y=230)
        
        
        self.stylebookbtn=ttk.Button(win, text='Open Stylebook', command=lambda:self.open_stylebook(win))
        self.stylebookbtn.place(x=75, y=350)
        self.set_style(win, 0, None)
        
        ttk.Label(win, text='Font Color',font=('Helvetica bold',16)).place(x=80, y=400)
        self.choose_color(win, False)
        ttk.Button(win,text='Change Color', command = lambda:self.choose_color(win)).place(x=76,y=550)
        
        ttk.Label(win, text='Text Orientation',font=('Helvetica bold',16)).place(x=355, y=20)
        options=['Left','Left', 'Middle', 'Right']
        self.orientation = tk.StringVar()
        self.orientation.set('Left')
        orientation_menu = ttk.OptionMenu(win, self.orientation, *options)
        orientation_menu.place(x=390,y=60)
        
        self.linewidthscale = ttk.Scale(win, length=200, from_=10, to=75, orient=tk.HORIZONTAL, value=42.5)# tickinterval=1)
        self.linewidthscale.place(x=332, y=195)
        
        ttk.Label(win, text='Max Line Width',font=('Helvetica bold',16)).place(x=357, y=120)
        ttk.Label(win, text='Narrow (10 chars)').place(x=285, y=152)
        ttk.Label(win, text='Wide (75 chars)').place(x=490, y=152)
        
        self.lineheightscale = ttk.Scale(win, length=200, from_=20, to=140, orient=tk.HORIZONTAL, value=80)# tickinterval=1)
        self.lineheightscale.place(x=332, y=322)
        
        ttk.Label(win, text='Line Spacing',font=('Helvetica bold',16)).place(x=357, y=250)
        ttk.Label(win, text='Narrow (20)').place(x=295, y=282)
        ttk.Label(win, text='Wide (140)').place(x=500, y=282)
        
        self.inputtext = tk.Text(win)
        scroll = ttk.Scrollbar(win) 
        self.inputtext.configure(yscrollcommand=scroll.set) 
        self.inputtext.pack(side=tk.LEFT) 
        scroll.config(command=self.inputtext.yview) 
        scroll.place(x=1070,y=20, height=450) 
        
        self.inputtext.place(x=600,y=20, width=470, height=450)
        
        largetext = ttk.Style()
        largetext.configure('my.TButton', font=('Helvetica', 26))
        self.generatebutton = ttk.Button(win, text='Generate Writing!',style='my.TButton', command=lambda:self.generate_dialog(win))
        self.generatebutton.place(x=680,y=510)
    
    def disallow_closing(self, root):
        #root.destroy()
        return
    
    def generate_dialog(self, win):
        self.generatebutton.config(state='disabled')
        self.text = self.inputtext.get("1.0", tk.END).replace('Q', 'q')
        print(self.text)
        if len("".join(self.text.split())) == 0:
            print('No text inputted!')
            messagebox.showerror('No Text!', 'Error: There is no text in the textbox.')
            self.generatebutton.config(state='normal')
            return
        
        char_array = list(self.text)
        for letter in char_array:
            if not letter in self.valid_chars:
                print('Invalid Char!')
                messagebox.showerror('Invalid Chars!', 'Error: Invalid Symbol Found: \"'+letter+'\"\nValid Symbols: a-z | A-Z | 0-9 | )(#"?\'.-.:;')
                self.generatebutton.config(state='normal')
                return
        
        self.generatedialog=tk.Toplevel(win)
        self.generatedialog.geometry("550x70")
        self.generatedialog.iconbitmap(resourcepath('gui/icon.ico'))
        self.generatedialog.title("Generating Writing!")
        self.generatedialog.resizable(0,0)
        self.generatedialog.protocol("WM_DELETE_WINDOW", lambda:self.disallow_closing(self.generatedialog))

        frame=ttk.Frame(self.generatedialog, width=550, height=70)
        frame.grid(row=0, column=0, sticky="NW")
        self.progress = ttk.Progressbar(self.generatedialog)
        self.generatelabel = ttk.Label(self.generatedialog, text='Initializing Hand')
        
        self.progress.place(relx=0.5, rely=0.35, anchor=tk.CENTER, width=500)
        self.generatelabel.place(relx=0.5, rely=0.6, anchor=tk.CENTER, width=500)
        
        t = Thread(target=self.generate_writing)
        t.setDaemon(True)
        t.start()
        
    def step_bar(self, bar, amount):
        if amount >= 1:
            wholeamount = math.floor(amount)
            for i in range(wholeamount):
                bar.step(1)
                time.sleep(.05/wholeamount)
            bar.step(amount-wholeamount)
        else:
            bar.step(amount)
            
    def split_string(self, text, length) -> list:
        words = text.split()
        result = []
        current = ''
        for word in words:
            if len(word) > length:
                result.append(word[:length])
                word = word[length:]
            if len(current + word) > length:
                result.append(current.strip())
                current = ''
            current += word + ' '
        result.append(current.strip())
        final = []
        for item in result:
            if not item == '':
                final.append(item)
        return final
   
    def generate_writing(self):
        print('Writing Started')
        if not self.hand:
            self.hand = Hand()
        Thread(target=lambda: self.step_bar(self.progress, 10)).start()
        
        self.generatelabel.config(text='Preparing Lines')
        
        
        bias = math.sqrt(self.legibilityscale.get()/100)
        width = self.widthscale.get()/4 + 0.6
        
        lines = self.text.splitlines()
        
        
        
        if os.path.isdir(resourcepath('handsynth-temp')):
            shutil.rmtree(resourcepath('handsynth-temp'))
        os.mkdir(resourcepath('handsynth-temp'))
        
        sublines = []
        synthesized_lines = []
        for line in lines:
            if len(line) > round(self.linewidthscale.get()):
                sublines = self.split_string(line, round(self.linewidthscale.get()))
                for subline in sublines:
                    synthesized_lines.append(subline)
                
            else:
                if not line == '':
                    synthesized_lines.append(line)
                else:
                    synthesized_lines.append(' ')
                
        proglength = 60/len(synthesized_lines)
        time.sleep(0.5)
        Thread(target=lambda: self.step_bar(self.progress, 5)).start()
        for i, line in enumerate(synthesized_lines):
            self.generatelabel.config(text='Synthesizing Lines - Please Wait (' + str(i+1) + '/' + str(len(synthesized_lines)) + ')')
            
            self.hand.write(
            filename=resourcepath(f'handsynth-temp/{i}.svg'),
            lines=[line],
            biases=[bias],
            styles=[self.currentstyle],
            stroke_colors=[self.stroke_color],
            stroke_widths=[width],
            )
            
            t = Thread(target=lambda: self.step_bar(self.progress, proglength))
            t.setDaemon(True)
            t.start()
            
        self.generatelabel.config(text='Rasterizing Lines')
        proglength = 20/len(synthesized_lines)
        
        for i in range(len(synthesized_lines)):
            self.generatelabel.config(text=f'Rasterizing Lines({i+1}/{len(synthesized_lines)})')
            if synthesized_lines[i] == ' ':
                Image.new('RGBA',(2_000 ,120)).save(resourcepath(f'handsynth-temp/{i}.png'))
            else:
                svg2png(resourcepath(f'handsynth-temp/{i}.svg'), resourcepath(f'handsynth-temp/{i}.png'))
            Thread(target=lambda: self.step_bar(self.progress, proglength)).start()
        
        line_spacing = round(self.lineheightscale.get()) 
        
        self.generatelabel.config(text='Combining Lines')
        canvas = Image.new('RGBA', (2_400, 400+round(line_spacing*len(synthesized_lines))))
        
        for i in range(len(synthesized_lines)):
            i+=1
            line_image = Image.open(resourcepath(f'handsynth-temp/{i-1}.png'))
            _, _, _, mask = line_image.split()
            if self.orientation.get() == 'Left':
                canvas.paste(line_image, (400,line_spacing*i), mask)
            elif self.orientation.get() == 'Right':
                canvas.paste(line_image, (2_000-line_image.width,line_spacing*i), mask)
            elif self.orientation.get() == 'Middle':
                canvas.paste(line_image, (1_000-(line_image.width/2),line_spacing*i), mask)
        canvas = fulltrim(canvas)
        finalcanvas = Image.new('RGBA', (canvas.width+120, canvas.height+120))
        finalcanvas.paste(canvas, (60,60))
        
        whitecanvas = Image.new('RGB', (canvas.width+120, canvas.height+120),color='white')
        _, _, _, mask = finalcanvas.split()
        whitecanvas.paste(finalcanvas, (0,0),mask=mask)
        
        self.step_bar(self.progress,4.9)
        self.generatebutton.config(state='normal')
        finalcanvas.show()
        if not os.path.isdir('outputs'):
            os.mkdir('outputs')
        
        fileid = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
        finalcanvas.save(f'outputs/{fileid}-alpha.png')
        whitecanvas.save(f'outputs/{fileid}-white.png')
        self.show_file(os.path.abspath(f'outputs/{fileid}-white.png'))
        shutil.rmtree(resourcepath('handsynth-temp'))
        time.sleep(.2)
        self.generatedialog.destroy()
    
    def show_file(self, file_path):
        if os.name == "nt":
            # For Windows
            subprocess.Popen(f'explorer /select,"{file_path}"')
        elif os.name == "posix":
            # For Linux and macOS
            subprocess.Popen(["xdg-open", file_path])
            
    def choose_color(self, win, usercolor=True):
        
        if usercolor:
            colorpickercolor = askcolor(self.stroke_color, win)[1]
            if colorpickercolor:
                self.stroke_color = colorpickercolor
                print(f'Color Chosen: {self.stroke_color}')
            else:
                return
            
        if self.colorlabel:
            self.colorlabel.destroy()
            
        if self.colortextlabel:
            self.colortextlabel.destroy()
            
        self.colorlabel = tk.Label(win, background=self.stroke_color,borderwidth=3, relief="sunken", width=31, height=4)
        self.colorlabel.place(x=20, y=440)
        self.colortextlabel = tk.Text(win,fg=self.stroke_color, bg='#ffffff')
        self.colortextlabel.insert(1.0,self.stroke_color)
        self.colortextlabel.configure(state="disabled")
        self.colortextlabel.place(x=100, y=520, width=55, height=22)
        

        
    def set_style(self, win, styleindex, stylebook):
        if stylebook:
            stylebook.destroy()
            
        if self.stylelabel:
            self.stylelabel.destroy()
            
        self.currentstyle = styleindex
        styleimage = ImageTk.PhotoImage(self.styles[styleindex])
        self.stylelabel = ttk.Label(image=styleimage, borderwidth=3, relief="sunken")
        self.stylelabel.image = styleimage
        self.stylelabel.place(x=20,y=270)
        
        
        
    def open_stylebook(self, win):
        buttons = []
        stylebook=tk.Toplevel(win)
        stylebook.resizable(0,0)
        stylebook.geometry("685x273")
        stylebook.title("Pick a style!")
        stylebook.iconbitmap(resourcepath('gui/icon.ico'))
        for x in range(4):
            for y in range(3):
                frame = tk.Frame(
                    master=stylebook,
                    relief=tk.RAISED,
                    borderwidth=1
                    )
                frame.grid(row=x, column=y)  # line 13
                styleindex = 3*x+y
                styleimage = ImageTk.PhotoImage(self.styles[styleindex])
                buttons.append(tk.Button(master=frame, text=f"\n\nrow {x}\t\t column {y}\n\n", image=styleimage, command=lambda index=styleindex:self.set_style(win, index, stylebook)))
                buttons[styleindex].image = styleimage
                buttons[styleindex].pack()

if __name__ == '__main__':
    window=tk.Tk()
    window.call('source', resourcepath('gui/azure/azure.tcl'))
    window.call('set_theme', 'dark')
    window.iconbitmap(resourcepath('gui/icon.ico'))
    window.resizable(0,0)
    mywin=MyWindow(window)
    window.title('Handwriting Synthesis')
    window.geometry("1100x600+10+10")
    window.mainloop()