import os
import ast
import spacy
import numpy as np
from errno import ENOENT
from collections import Counter

nlp = spacy.load("en")
