'''
templates and label mappings
'''

t1_fine = {
    # [MASK] [P] [A] [P] [S] [P]
    'content': ['[CLS] [MASK]', '[SEP]'],
    'map': [0, 'p', 'a', 'p', 's', 'p', 1],

    # default multi-modal template
    'default': {
        # [Img] [P] [MASK] [P] [A] [P] [S] [P]
        'content': ['[CLS]', '[MASK]', '[SEP]'],
        'map': [0, 'i', 'p', 1, 'p', 'a', 'p', 's', 'p', 2]
    },
}

t2_fine = {
    'content': ['[CLS] Text : " ', ' " . Aspect: " ', ' " . Sentiment of aspect : [MASK] . [SEP]'],
    'map': [0, 's', 1, 'a', 2],

    # default multi-modal template
    'default': {
        'content': ['[CLS]', '[SEP] Text : " ', ' " . Aspect: " ', ' " . Sentiment of aspect : [MASK] . [SEP]'],
        'map': [0, 'i', 1, 's', 2, 'a', 3],
    },
}

template_fine = {
    1: t1_fine,
    2: t2_fine,
}


t1_coarse = {
    # [MASK] [P] [S] [P]
    'content': ['[CLS] [MASK]', '[SEP]'],
    'map': [0, 'p', 's', 'p', 1],

    # default multi-modal template
    'default': {
        # [Img] [P] [MASK] [P] [S] [P]
        'content': ['[CLS]', '[MASK]', '[SEP]'],
        'map': [0, 'i', 'p', 1, 'p', 's', 'p', 2],
    },
}

t2_coarse = {
    'content': ['[CLS] Text : " ', ' " . Sentiment of text : [MASK] . [SEP]'],
    'map': [0, 's', 1],

    # default multi-modal template
    'default': {
        'content': ['[CLS]', '[SEP] Text : " ', ' " . Sentiment of text : [MASK] . [SEP]'],
        'map': [0, 'i', 1, 's', 2],
    },
}

template_coarse = {
    1: t1_coarse,
    2: t2_coarse,
}


def twitter(template: int):
    label_list = ['negative', 'neutral', 'positive']
    label_map = {'0': "negative", '1': "neutral", '2': "positive"}
    return label_list, label_map, template_fine[template]


def masad(template: int):
    label_list = ['negative', 'positive']
    label_map = {'negative': "negative", 'positive': "positive"}
    return label_list, label_map, template_fine[template]


def mvsa(template: int):
    label_list = ['negative', 'neutral', 'positive']
    label_map = {'negative': "negative", 'neutral': "neutral", 'positive': "positive"}
    return label_list, label_map, template_coarse[template]


def tumemo(template: int):
    label_list = ['angry', 'bored', 'calm', 'fear', 'happy', 'love', 'sad']
    label_map = {'Angry': 'angry', 'Bored': 'bored', 'Calm': 'calm', 'Fear': 'fear', 'Happy': 'happy', 'Love': 'love', 'Sad': 'sad'}
    return label_list, label_map, template_coarse[template]


processors = {
    't2015': twitter,
    't2017': twitter,
    'masad': masad,
    'mvsa-s': mvsa,
    'mvsa-m': mvsa,
    'tumemo': tumemo,
}
