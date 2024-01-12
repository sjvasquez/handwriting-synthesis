# Copyright (c) 2021 rdbende <rdbende@gmail.com>

# The Azure theme is a beautiful modern ttk theme inspired by Microsoft's fluent design.

package require Tk 8.6

namespace eval ttk::theme::azure-dark {
    variable version 2.0
    package provide ttk::theme::azure-dark $version

    ttk::style theme create azure-dark -parent clam -settings {
        proc load_images {imgdir} {
            variable I
            foreach file [glob -directory $imgdir *.png] {
                set img [file tail [file rootname $file]]
                set I($img) [image create photo -file $file -format png]
            }
        }

        load_images [file join [file dirname [info script]] dark]

        array set colors {
            -fg             "#ffffff"
            -bg             "#333333"
            -disabledfg     "#aaaaaa"
            -disabledbg     "#737373"
            -selectfg       "#ffffff"
            -selectbg       "#007fff"
        }

        ttk::style layout TButton {
            Button.button -children {
                Button.padding -children {
                    Button.label -side left -expand true
                } 
            }
        }

        ttk::style layout Toolbutton {
            Toolbutton.button -children {
                Toolbutton.padding -children {
                    Toolbutton.label -side left -expand true
                } 
            }
        }

        ttk::style layout TMenubutton {
            Menubutton.button -children {
                Menubutton.padding -children {
                    Menubutton.indicator -side right
                    Menubutton.label -side right -expand true
                }
            }
        }

        ttk::style layout TOptionMenu {
            OptionMenu.button -children {
                OptionMenu.padding -children {
                    OptionMenu.indicator -side right
                    OptionMenu.label -side right -expand true
                }
            }
        }

        ttk::style layout Accent.TButton {
            AccentButton.button -children {
                AccentButton.padding -children {
                    AccentButton.label -side left -expand true
                } 
            }
        }

        ttk::style layout TCheckbutton {
            Checkbutton.button -children {
                Checkbutton.padding -children {
                    Checkbutton.indicator -side left
                    Checkbutton.label -side right -expand true
                }
            }
        }

        ttk::style layout Switch.TCheckbutton {
            Switch.button -children {
                Switch.padding -children {
                    Switch.indicator -side left
                    Switch.label -side right -expand true
                }
            }
        }

        ttk::style layout Toggle.TButton {
            ToggleButton.button -children {
                ToggleButton.padding -children {
                    ToggleButton.label -side left -expand true
                } 
            }
        }

        ttk::style layout TRadiobutton {
            Radiobutton.button -children {
                Radiobutton.padding -children {
                    Radiobutton.indicator -side left
                    Radiobutton.label -side right -expand true
                }
            }
        }

        ttk::style layout Vertical.TScrollbar {
            Vertical.Scrollbar.trough -sticky ns -children {
                Vertical.Scrollbar.thumb -expand true
            }
        }

        ttk::style layout Horizontal.TScrollbar {
            Horizontal.Scrollbar.trough -sticky ew -children {
                Horizontal.Scrollbar.thumb -expand true
            }
        }

        ttk::style layout TCombobox {
            Combobox.field -sticky nswe -children {
                Combobox.padding -expand true -sticky nswe -children {
                    Combobox.textarea -sticky nswe
                }
            }
            Combobox.button -side right -sticky ns -children {
                Combobox.arrow -sticky nsew
            }
        }

        ttk::style layout TSpinbox {
            Spinbox.field -sticky nsew -children {
                Spinbox.padding -expand true -sticky nswe -children {
                    Spinbox.textarea -sticky nswe
                }
                
            }
            Spinbox.button -side right -sticky ns -children {
                null -side right -children {
                    Spinbox.uparrow -side top
                    Spinbox.downarrow -side bottom
                }
            }
        }

        ttk::style layout Horizontal.TSeparator {
            Horizontal.separator -sticky nswe
        }

        ttk::style layout Vertical.TSeparator {
            Vertical.separator -sticky nswe
        }
        
        ttk::style layout Horizontal.Tick.TScale {
            Horizontal.TickScale.trough -sticky ew -children {
                Horizontal.TickScale.slider -sticky w
            }
        }
        
        ttk::style layout Vertical.Tick.TScale {
            Vertical.TickScale.trough -sticky ns -children {
                Vertical.TickScale.slider -sticky n
            }
        }

        ttk::style layout Card.TFrame {
            Card.field {
                Card.padding -expand 1 
            }
        }

        ttk::style layout TLabelframe {
            Labelframe.border {
                Labelframe.padding -expand 1 -children {
                    Labelframe.label -side right
                }
            }
        }

        ttk::style layout TNotebook.Tab {
            Notebook.tab -children {
                Notebook.padding -side top -children {
                    Notebook.label -side top -sticky {}
                }
            }
        }

        ttk::style layout Treeview.Item {
            Treeitem.padding -sticky nswe -children {
                Treeitem.indicator -side left -sticky {}
                Treeitem.image -side left -sticky {}
                Treeitem.text -side left -sticky {}
            }
        }


        # Elements

        # Button
        ttk::style configure TButton -padding {8 4 8 4} -width -10 -anchor center

        ttk::style element create Button.button image \
            [list $I(rect-basic) \
            	{selected disabled} $I(rect-basic) \
                disabled $I(rect-basic) \
                pressed $I(rect-basic) \
                selected $I(rect-basic) \
                active $I(button-hover) \
                focus $I(button-hover) \
            ] -border 4 -sticky ewns

        # Toolbutton
        ttk::style configure Toolbutton -padding {8 4 8 4} -width -10 -anchor center

        ttk::style element create Toolbutton.button image \
            [list $I(empty) \
            	{selected disabled} $I(empty) \
                disabled $I(empty) \
                pressed $I(rect-basic) \
                selected $I(rect-basic) \
                active $I(rect-basic) \
            ] -border 4 -sticky ewns

        # Menubutton
        ttk::style configure TMenubutton -padding {8 4 4 4}

        ttk::style element create Menubutton.button \
            image [list $I(rect-basic) \
                disabled $I(rect-basic) \
                pressed $I(rect-basic) \
                active $I(button-hover) \
            ] -border 4 -sticky ewns 

        ttk::style element create Menubutton.indicator \
            image [list $I(down) \
                active   $I(down) \
                pressed  $I(down) \
                disabled $I(down) \
            ] -width 15 -sticky e

        # OptionMenu
        ttk::style configure TOptionMenu -padding {8 4 4 4}

        ttk::style element create OptionMenu.button \
            image [list $I(rect-basic) \
                disabled $I(rect-basic) \
                pressed $I(rect-basic) \
                active $I(button-hover) \
            ] -border 4 -sticky ewns 

        ttk::style element create OptionMenu.indicator \
            image [list $I(down) \
                active   $I(down) \
                pressed  $I(down) \
                disabled $I(down) \
            ] -width 15 -sticky e

        # AccentButton
        ttk::style configure Accent.TButton -padding {8 4 8 4} -width -10 -anchor center

        ttk::style element create AccentButton.button image \
            [list $I(rect-accent) \
            	{selected disabled} $I(rect-accent-hover) \
                disabled $I(rect-accent-hover) \
                pressed $I(rect-accent) \
                selected $I(rect-accent) \
                active $I(rect-accent-hover) \
                focus $I(rect-accent-hover) \
            ] -border 4 -sticky ewns

        # Checkbutton
        ttk::style configure TCheckbutton -padding 4

        ttk::style element create Checkbutton.indicator image \
            [list $I(box-basic) \
                {alternate disabled} $I(check-tri-basic) \
                {selected disabled} $I(check-basic) \
                disabled $I(box-basic) \
                {pressed alternate} $I(check-tri-hover) \
                {active alternate} $I(check-tri-hover) \
                alternate $I(check-tri-accent) \
                {pressed selected} $I(check-hover) \
                {active selected} $I(check-hover) \
                selected $I(check-accent) \
                {pressed !selected} $I(rect-hover) \
                active $I(box-hover) \
            ] -width 26 -sticky w

        # Switch
        ttk::style element create Switch.indicator image \
            [list $I(off-basic) \
                {selected disabled} $I(on-basic) \
                disabled $I(off-basic) \
                {pressed selected} $I(on-accent) \
                {active selected} $I(on-accent) \
                selected $I(on-accent) \
                {pressed !selected} $I(off-basic) \
                active $I(off-basic) \
            ] -width 46 -sticky w

        # ToggleButton
        ttk::style configure Toggle.TButton -padding {8 4 8 4} -width -10 -anchor center

        ttk::style element create ToggleButton.button image \
            [list $I(rect-basic) \
                {selected disabled} $I(rect-accent-hover) \
                disabled $I(rect-basic) \
                {pressed selected} $I(rect-basic) \
                {active selected} $I(rect-accent) \
                selected $I(rect-accent) \
                {pressed !selected} $I(rect-accent) \
                active $I(rect-basic) \
            ] -border 4 -sticky ewns

        # Radiobutton
        ttk::style configure TRadiobutton -padding 4

        ttk::style element create Radiobutton.indicator image \
            [list $I(outline-basic) \
                {alternate disabled} $I(radio-tri-basic) \
                {selected disabled} $I(radio-basic) \
                disabled $I(outline-basic) \
                {pressed alternate} $I(radio-tri-hover) \
                {active alternate} $I(radio-tri-hover) \
                alternate $I(radio-tri-accent) \
                {pressed selected} $I(radio-hover) \
                {active selected} $I(radio-hover) \
                selected $I(radio-accent) \
                {pressed !selected} $I(circle-hover) \
                active $I(outline-hover) \
            ] -width 26 -sticky w

        # Scrollbar
        ttk::style element create Horizontal.Scrollbar.trough image $I(hor-basic) \
            -sticky ew

        ttk::style element create Horizontal.Scrollbar.thumb \
             image [list $I(hor-accent) \
                disabled $I(hor-basic) \
                pressed $I(hor-hover) \
                active $I(hor-hover) \
            ] -sticky ew

        ttk::style element create Vertical.Scrollbar.trough image $I(vert-basic) \
            -sticky ns

        ttk::style element create Vertical.Scrollbar.thumb \
            image [list $I(vert-accent) \
                disabled  $I(vert-basic) \
                pressed $I(vert-hover) \
                active $I(vert-hover) \
            ] -sticky ns

        # Scale
        ttk::style element create Horizontal.Scale.trough image $I(scale-hor) \
            -border 5 -padding 0

        ttk::style element create Horizontal.Scale.slider \
            image [list $I(circle-accent) \
                disabled $I(circle-basic) \
                pressed $I(circle-hover) \
                active $I(circle-hover) \
            ] -sticky {}

        ttk::style element create Vertical.Scale.trough image $I(scale-vert) \
            -border 5 -padding 0

        ttk::style element create Vertical.Scale.slider \
            image [list $I(circle-accent) \
                disabled $I(circle-basic) \
                pressed $I(circle-hover) \
                active $I(circle-hover) \
            ] -sticky {}
            
        # Tickscale
        ttk::style element create Horizontal.TickScale.trough image $I(scale-hor) \
            -border 5 -padding 0
        
        ttk::style element create Horizontal.TickScale.slider \
            image [list $I(tick-hor-accent) \
                disabled $I(tick-hor-basic) \
                pressed $I(tick-hor-hover) \
                active $I(tick-hor-hover) \
            ] -sticky {}
            
        ttk::style element create Vertical.TickScale.trough image $I(scale-vert) \
            -border 5 -padding 0

        ttk::style element create Vertical.TickScale.slider \
            image [list $I(tick-vert-accent) \
                disabled $I(tick-vert-basic) \
                pressed $I(tick-vert-hover) \
                active $I(tick-vert-hover) \
            ] -sticky {}

        # Progressbar
        ttk::style element create Horizontal.Progressbar.trough image $I(hor-basic) \
            -sticky ew

        ttk::style element create Horizontal.Progressbar.pbar image $I(hor-accent) \
            -sticky ew

        ttk::style element create Vertical.Progressbar.trough image $I(vert-basic) \
            -sticky ns

        ttk::style element create Vertical.Progressbar.pbar image $I(vert-accent) \
            -sticky ns

        # Entry
        ttk::style element create Entry.field \
            image [list $I(box-basic) \
                {focus hover} $I(box-accent) \
                invalid $I(box-invalid) \
                disabled $I(box-basic) \
                focus $I(box-accent) \
                hover $I(box-hover) \
            ] -border 5 -padding {8} -sticky news

        # Combobox
        ttk::style map TCombobox -selectbackground [list \
            {!focus} $colors(-selectbg) \
            {readonly hover} $colors(-selectbg) \
            {readonly focus} $colors(-selectbg) \
        ]
            
        ttk::style map TCombobox -selectforeground [list \
            {!focus} $colors(-selectfg) \
            {readonly hover} $colors(-selectfg) \
            {readonly focus} $colors(-selectfg) \
        ]

        ttk::style element create Combobox.field \
            image [list $I(box-basic) \
                {readonly disabled} $I(rect-basic) \
                {readonly pressed} $I(rect-basic) \
                {readonly focus hover} $I(button-hover) \
                {readonly focus} $I(button-hover) \
                {readonly hover} $I(button-hover) \
                {focus hover} $I(box-accent) \
                readonly $I(rect-basic) \
                invalid $I(box-invalid) \
                disabled $I(box-basic) \
                focus $I(box-accent) \
                hover $I(box-hover) \
            ] -border 5 -padding {8}
            
        ttk::style element create Combobox.button \
            image [list $I(combo-button-basic) \
                 {!readonly focus} $I(combo-button-focus) \
                 {readonly focus} $I(combo-button-hover) \
                 {readonly hover} $I(combo-button-hover)
            ] -border 5 -padding {2 6 6 6}

        ttk::style element create Combobox.arrow image $I(down) \
            -width 15 -sticky e

        # Spinbox
        ttk::style element create Spinbox.field \
            image [list $I(box-basic) \
                invalid $I(box-invalid) \
                disabled $I(box-basic) \
                focus $I(box-accent) \
                hover $I(box-hover) \
            ] -border 5 -padding {8} -sticky news

        ttk::style element create Spinbox.uparrow \
            image [list $I(up) \
                disabled $I(up) \
                pressed $I(up-accent) \
                active $I(up-accent) \
            ] -border 4 -width 15 -sticky e

        ttk::style element create Spinbox.downarrow \
            image [list $I(down) \
                disabled $I(down) \
                pressed $I(down-accent) \
                active $I(down-accent) \
            ] -border 4 -width 15 -sticky e
            
        ttk::style element create Spinbox.button \
            image [list $I(combo-button-basic) \
                 {!readonly focus} $I(combo-button-focus) \
                 {readonly focus} $I(combo-button-hover) \
                 {readonly hover} $I(combo-button-hover)
            ] -border 5 -padding {2 6 6 6}

        # Sizegrip
        ttk::style element create Sizegrip.sizegrip image $I(size) \
            -sticky ewns

        # Separator
        ttk::style element create Horizontal.separator image $I(separator)

        ttk::style element create Vertical.separator image $I(separator)
        
        # Card
        ttk::style element create Card.field image $I(card) \
            -border 10 -padding 4 -sticky news

        # Labelframe
        ttk::style element create Labelframe.border image $I(card) \
            -border 5 -padding 4 -sticky news
        
        # Notebook
        ttk::style element create Notebook.client \
            image $I(notebook) -border 5

        ttk::style element create Notebook.tab \
            image [list $I(tab-disabled) \
                selected $I(tab-basic) \
                active $I(tab-hover) \
            ] -border 5 -padding {14 4}

        # Treeview
        ttk::style element create Treeview.field image $I(card) \
            -border 5

        ttk::style element create Treeheading.cell \
            image [list $I(tree-basic) \
                pressed $I(tree-pressed)
            ] -border 5 -padding 4 -sticky ewns
        
        ttk::style element create Treeitem.indicator \
            image [list $I(right) \
                user2 $I(empty) \
                user1 $I(down) \
            ] -width 26 -sticky {}

        ttk::style configure Treeview -background $colors(-bg)
        ttk::style configure Treeview.Item -padding {2 0 0 0}
        ttk::style map Treeview \
            -background [list selected $colors(-selectbg)] \
            -foreground [list selected $colors(-selectfg)]

        # Panedwindow
        # Insane hack to remove clam's ugly sash
        ttk::style configure Sash -gripcount 0
    }
}
