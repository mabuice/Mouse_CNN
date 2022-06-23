=========
Mouse_CNN
=========
This is the repo for the ongoing project of CNN MouseNet -- a convolutional neural network constrained by the architecture of the mouse visual cortex. 

.. contents:: Table of Contents
   :depth: 2

Folder Structure
================

::

  Mouse_CNN/
  │
  ├── mousenet/
  │    │
  │    │
  │    ├── cmouse/ - Code related to constructing the PyTorch model
  │    │   └── __init__.py 
  │    │
  │    ├── example/ - Example code and resources
  │    │
  │    ├── mouse_cnn/ - Code related to deriving architecture from data
  │    │   └── __init__.py
  │    │
  │    ├── retinotopics/ - Code related to calculating visual subfields
  │    │   └── __init__.py
  │    │
  │    └── /
  │        ├── loader.py - load function for loading a mousenet model with a particular initialization
  │        └── __init__.py - manages pathing for saving models + logs
  │
  ├── environment.yml - Conda environment and dependancies
  │
  ├── setup.cfg - Development configurations and linting
  │
  ├── setup.py - package definitions
  │
  └── tests/ - tests folder


Usage
=====

Installation: 
Change directory to cloned folder

.. code-block::
   $ pip install -e . 



To load a mousenet model

.. code-block::

  $ import mousenet
  $ model = mousenet.load(architecture="stock", pretraining=None)
  
Architecture can be one of "stock" or "retinotopic" for visual subfields. Pretraining can be on of: None, "kaiming" for kaiming initialization, or "Imagenet" for imagenet pretraining.


To test the code

.. code-block::

   $ pytest
