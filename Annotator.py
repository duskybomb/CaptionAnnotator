import random
import functools
import pandas as pd
from IPython.display import display, clear_output
from ipywidgets import Button, Dropdown, HTML, HBox, IntSlider, FloatSlider, Textarea, Output
from sklearn.model_selection import train_test_split
from PIL import Image as PILImage
import ipyplot


class Annotator:
    def __init__(self, examples, options=None, shuffle=False, include_skip=False, max_options_dropdown=10, 
                        show_columns=None, annotation_column_name='label', display_fn=display):

        self.options = options
        self.shuffle = shuffle
        self.include_skip = include_skip
        self.max_options_dropdown = max_options_dropdown
        self.show_columns = show_columns
        self.annotation_column_name = annotation_column_name
        self.display_fn = display
        self.current_index = -1
        self.current_train_score = 0.0
        self.current_test_score = 0.0

        self.examples = examples.copy()

        
        self.show_columns = ['images']

        if annotation_column_name not in self.examples.columns:
            self.examples[annotation_column_name] = None
            self.examples["caption"] = None

        self.pipeline = None

        task_type = 'classification'
       
        self.task_type = task_type
        
    def add_pipeline(self, pipeline, fit_every_n_sample=10, fit_minimum_samples=50, fit_test_size=0.2, fit_features='all'):
        self.pipeline = pipeline
        self.fit_every_n_sample = fit_every_n_sample
        self.fit_minimum_samples = fit_minimum_samples
        self.fit_test_size = fit_test_size
        self.fit_features = [f for f in fit_features if f in self.examples.columns]

    def init_html(self):
        self.count_label = HTML()


    def set_label_text(self):

        self.count_label.value = '{} examples annotated, {} examples left'.format(
            len(self.examples) - self.examples[self.annotation_column_name].isnull().sum(), 
            len(self.examples) - self.current_index
        )

    def show_next(self):
        self.current_index += 1
        self.set_label_text()
        if self.current_index >= len(self.examples):
            for btn in self.buttons:
                btn.disabled = True
            print('Annotation done.')
            return

            #current index should be len(examples.dropna(subset=['label']))
        with self.out:
            clear_output(wait=True)
            display(f"Title: {self.examples.loc[self.current_index]['title']}"),
            display(f"Description: {self.examples.loc[self.current_index]['description']}")
            display(f"Original Label: {self.examples.loc[self.current_index]['reported_category']}")
            display(f"Structured Label: {self.examples.loc[self.current_index]['category']}")
            # display(f"Lablled Adit Title: {self.examples.loc[self.current_index]['title_adit']}")
            ipyplot.plot_images(self.examples.loc[[self.current_index]][self.show_columns[0]][self.current_index], max_images=20, img_width=150)
            

    def add_annotation(self, cat, caption):
        if cat == "--Select--":
            cat = self.examples.loc[self.current_index]['category']
        if caption == "":
            caption = self.examples.loc[self.current_index]['title_adit']
        self.examples.iloc[self.current_index, -1] = caption
        self.examples.iloc[self.current_index, -2] = cat
        self.show_next()

    def skip(self, btn):
        self.show_next()
    

    def annotate(self):
        self.count_label = HTML()
        self.set_label_text()
        display(self.count_label)
        self.buttons = []
        text_id = self.current_index+1

        if self.task_type == 'classification':
            use_dropdown = len(self.options) > self.max_options_dropdown

            if use_dropdown:
                dd = Dropdown(options=self.options)
                display(dd)
            
            else:
                for label in self.options:
                    btn = Button(description=label)
                    def on_click(label, btn):
                        self.add_annotation(label)
                    btn.on_click(functools.partial(on_click, label))
                    self.buttons.append(btn)

        elif self.task_type == 'regression':
            target_type = type(options[0])
            if target_type == int:
                cls = IntSlider
            else:
                cls = FloatSlider
            if len(options) == 2:
                min_val, max_val = options
                slider = cls(min=min_val, max=max_val)
            else:
                min_val, max_val, step_val = options
                slider = cls(min=min_val, max=max_val, step=step_val)
            display(slider)
            btn = Button(description='submit')
            def on_click(btn):
                add_annotation(slider.value)
            btn.on_click(on_click)
            buttons.append(btn)

        ta = Textarea()
        display(ta)
        btn = Button(description='submit', button_style="success")
        def on_click(btn):
            self.add_annotation(dd.value, ta.value)
        btn.on_click(on_click)
        self.buttons.append(btn)

        self.box = HBox(self.buttons)
        display(self.box)

        self.out = Output()
        display(self.out)

        self.show_next()

    def get_annotated_examples(self):
        return self.examples.dropna(subset=['caption'])[['link', 'label', 'caption']]