import re

PAD = 'PADPAD'
UNK = 'UNKUNK'
STA= 'BOSBOS'
END = 'EOSEOS'

PAD_ID = 0
UNK_ID = 1
STA_ID = 2
END_ID = 3

EMBED_DIM = 300
version = 'v1'

def reform_text(text):
    '''
    Removes unwanted characters from the given text
    '''
    text = re.sub(u'-|¢|¥|€|£|\u2010|\u2011|\u2012|\u2013|\u2014|\u2015|%|\[|\]|:|\(|\)|/',
                lambda text: ' ' + text.group(0) + ' ', text)
    text = text.strip(' \n')
    text = re.sub('\s+', ' ', text)
    return text
