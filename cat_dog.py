#!/usr/bin/env python
# coding: utf-8

# In[2]:


from fastai.vision.all import *


# In[3]:


path = untar_data(URLs.PETS)/'images'


# In[4]:


def is_cat(x): return x[0].isupper() 


# In[5]:


dls = ImageDataLoaders.from_name_func(       
    path,
    get_image_files(path),
    valid_pct=0.2,
    seed=42,
    label_func=is_cat,
    item_tfms=Resize(224))


# In[6]:


learn = cnn_learner(dls, resnet34, metrics=error_rate)        
learn.fine_tune(1) 


# In[1]:


from fastai.vision.all import *

# 1. Data Collection
path = untar_data(URLs.PETS)/'images'

# 2. Data Preprocessing
def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))

# 3. Model Selection
learn = cnn_learner(dls, resnet34, metrics=error_rate)

# 4. Model Training
learn.fine_tune(1)

# 5. Model Evaluation
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

# 6. Model Deployment (Example)
img = PILImage.create('cat.jpg')
is_cat, _, _ = learn.predict(img)
print(f"Is this a cat?: {is_cat}.")


# In[3]:


img = PILImage.create('cat.jpg')
is_cat, _, _ = learn.predict(img)
img.show()
print(f"Is this a cat?: {is_cat}.")

