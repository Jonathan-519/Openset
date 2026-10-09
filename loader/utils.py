"""Image decoding and historical taxonomy-name normalization."""
from PIL import Image


def default_loader(path):
    return Image.open(path).convert('RGB')


def prepro_node_name(x):
    x = [y for y in x if y not in '0123456789']
    if x[-1] == '_':
        x = x[:-1]
    return ''.join(x).replace('_', ' ')
