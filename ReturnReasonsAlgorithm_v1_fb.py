# ### IMPORTING DEPENDENCIES

import pandas as pd
import re
import numpy as np
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('max_colwidth', None)
from nltk.corpus import stopwords
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict
# nltk.download('stopwords')
from collections import defaultdict
from warnings import filterwarnings
filterwarnings('ignore')
import time
from transformers import pipeline
# classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


# ### DATA ADDITION

df = pd.read_csv('/Users/anubhavgupta/Desktop/Return_Reasons_Project/Python/Code/Return_Reasons_test_2024_02_26.csv')

df['rpi_count'] = df['count(rpi.sale_order_item_id)']
df.drop(columns='count(rpi.sale_order_item_id)', inplace=True)
df['short_rr'] = df['return_reason']

# ## Removing Null Values

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

source_codes = ['MYNTRAPPMP',
                'FLIPKART',
                'AMAZON_IN',
                'AMAZON_FLEX_API',
                'NYKAA_FASHION',
                'AJIO_OMNI',
                'MEESHO',
                'AJIO',
                'MYNTRA_B2B',
                'SNAPDEAL',
                'AMAZON_IN_API',
                'LIMEROAD',
                'FIRSTCRY',
                'TATA_CLIQ',
                'NYKAA_COM',
                'AMAZON_FBA_IN',
                'FLIPKART_FA',
                'AMAZON_FBA',
                'AMAZON_FLEX',
                'AMAZON_EASYSHIP',
                'CRED',
                'NYKAA',
                'JIOMART',
                'JIOMART3P']


df = df[df['source_code'].isin(source_codes)].reset_index(drop=True)
org_df = df.copy()

# ### Lowering Each Word 
return_reason = []
for reason in df['short_rr']:
    return_reason.append(reason.lower())
    
df['short_rr'] = return_reason
df['short_rr'] = df['short_rr'].apply(lambda x: x.replace("'", ""))
df['short_rr'] = df['short_rr'].apply(lambda x: x.replace("’", ""))

def clean_txt(text):
    return re.sub(r"[^a-z0-9]", " ", text)
df['short_rr'] = df['short_rr'].apply(clean_txt)
df['short_rr'] = df['short_rr'].str.replace(r'\s+', ' ', regex=True).str.strip()
df = df[df['short_rr'] != ""]
# # ROW REMOVAL 1x
# -- CONTAINS SPECFIC WORDS --

df = df[~df['short_rr'].str.contains('auto removed')]
df = df[~df['short_rr'].str.contains('rtom')]
df = df[~df['short_rr'].str.contains('http')]
df = df[~df['short_rr'].str.contains('trial')]
df = df[~df['short_rr'].str.contains('rtoa')]
df = df[~df['short_rr'].str.contains('return reason thanos roh approval flow')]
df = df[~df['short_rr'].str.contains('return expected on panel')]
df = df[~df['short_rr'].str.contains('test')]
df = df[~df['short_rr'].str.contains('myec')]
df = df[~df['short_rr'].str.contains('swit')]
df = df[~df['short_rr'].str.contains('ajio')]
df = df[~df['short_rr'].str.contains('origin')]
df = df[~df['short_rr'].str.contains('myn')]
df = df[~df['short_rr'].str.contains('limeroad')]
df = df[~df['short_rr'].str.contains('flex')]
df = df[~df['short_rr'].str.contains('nykaa')]
df = df[~df['short_rr'].str.contains('reason not available')]
df = df[~df['short_rr'].str.contains('others return reason')]
df = df[~df['short_rr'].str.contains(r'^received$')]
df = df[~df['short_rr'].str.contains('approved')]
df = df[~df['short_rr'].str.contains('pickup')]
df = df[~df['short_rr'].str.contains('address')]
df = df[~df['short_rr'].str.contains('manually')]
df = df[~df['short_rr'].str.contains('myer')]
df = df[~df['short_rr'].str.contains('crm')]
df = df[~df['short_rr'].str.contains('inventory')]

# -- CONTAINS ONLY WORDS --

df = df[df['short_rr'] != 'undefined']
df = df[df['short_rr'] != 'rto'] 
df = df[df['short_rr'] != 'courier return'] 
df = df[df['short_rr'] != 'return'] 
df = df[df['short_rr'] != 'cr'] 
df = df[df['short_rr'] != 'customer return']
df = df[df['short_rr'] != 'rvp']
df = df[df['short_rr'] != 'rtv']

# -- CONTAINS CERTAIN PATTERNS -- 

pattern4 = r'rto \d+$'
for text in df['short_rr']:
    matches = re.findall(pattern4, text)
matches = df[df['short_rr'].str.contains(pattern4)]['short_rr'].tolist()
df = df[~df['short_rr'].str.contains(pattern4)]

pattern4 = r'rto\d+$'
for text in df['short_rr']:
    matches = re.findall(pattern4, text)
matches = df[df['short_rr'].str.contains(pattern4)]['short_rr'].tolist()
df = df[~df['short_rr'].str.contains(pattern4)]

pattern4 = r'customer return \d+$'
for text in df['short_rr']:
    matches = re.findall(pattern4, text)
matches = df[df['short_rr'].str.contains(pattern4)]['short_rr'].tolist()
df = df[~df['short_rr'].str.contains(pattern4)]

pattern4 = r'customer return\d+$'
for text in df['short_rr']:
    matches = re.findall(pattern4, text)
matches = df[df['short_rr'].str.contains(pattern4)]['short_rr'].tolist()
df = df[~df['short_rr'].str.contains(pattern4)]

pattern4 = r'rtv \d+$'
for text in df['short_rr']:
    matches = re.findall(pattern4, text)
df = df[~df['short_rr'].str.contains(pattern4)]

pattern4 = r'rtv\d+$'
for text in df['short_rr']:
    matches = re.findall(pattern4, text)
df = df[~df['short_rr'].str.contains(pattern4)]

pattern4 = r'fmpc\d+$'
for text in df['short_rr']:
    matches = re.findall(pattern4, text)
df = df[~df['short_rr'].str.contains(pattern4)]

pattern4 = r'fmpc \d+$'
for text in df['short_rr']:
    matches = re.findall(pattern4, text)
df = df[~df['short_rr'].str.contains(pattern4)]

pattern4 = r'fmpr\d+$'
for text in df['short_rr']:
    matches = re.findall(pattern4, text)
df = df[~df['short_rr'].str.contains(pattern4)]

pattern4 = r'fmpr \d+$'
for text in df['short_rr']:
    matches = re.findall(pattern4, text)
df = df[~df['short_rr'].str.contains(pattern4)]

pattern4 = r'fmpp\d+$'
for text in df['short_rr']:
    matches = re.findall(pattern4, text)
df = df[~df['short_rr'].str.contains(pattern4)]

pattern4 = r'fmpp \d+$'
for text in df['short_rr']:
    matches = re.findall(pattern4, text)
df = df[~df['short_rr'].str.contains(pattern4)]

pattern4 = r'myep\d+$'
for text in df['short_rr']:
    matches = re.findall(pattern4, text)
df = df[~df['short_rr'].str.contains(pattern4)]

pattern4 = r'myep \d+$'
for text in df['short_rr']:
    matches = re.findall(pattern4, text)
df = df[~df['short_rr'].str.contains(pattern4)]

word_s = 'rtomanish'
df = df[~df['short_rr'].str.contains(word_s)]

word_s = ' ebo '
df = df[~df['short_rr'].str.contains(word_s)]

word_s = 'shipment bagout'
df = df[~df['short_rr'].str.contains(word_s)]

word_s = 'null null'
df = df[~df['short_rr'].str.contains(word_s)]

# -- WORDS REPLACEMENT --

df['short_rr'] = df['short_rr'].apply(lambda x: x.replace("recd", "received"))
df['short_rr'] = df['short_rr'].apply(lambda x: x.replace("damage", "damaged"))
df['short_rr'] = df['short_rr'].apply(lambda x: x.replace("damagedd", "damaged"))
df['short_rr'] = df['short_rr'].apply(lambda x: x.replace("colour", "color"))
df['short_rr'] = df['short_rr'].apply(lambda x: x.replace("issue", "issues"))
df['short_rr'] = df['short_rr'].apply(lambda x: x.replace("issuess", "issues"))
df['short_rr'] = df['short_rr'].apply(lambda x: x.replace("qc", "quality"))
df['short_rr'] = df['short_rr'].apply(lambda x: x.replace("cancelled", "cancel"))
df['short_rr'] = df['short_rr'].apply(lambda x: x.replace("use", "used"))
df['short_rr'] = df['short_rr'].apply(lambda x: x.replace("tag ", "tags "))
df['short_rr'] = df['short_rr'].apply(lambda x: x.replace("tag ", "tags "))
df['short_rr'] = df['short_rr'].apply(lambda x: x.replace("sku", ""))

# -- REMOVING WITH ONLY NUMBERS --

pattern = r'^[0-9-]+$'
for text in df['short_rr']:
    matches = re.findall(pattern, text)  
matches = df[df['short_rr'].str.contains(pattern)]['short_rr'].tolist()
df = df[~df['short_rr'].str.contains(pattern)]

# -- REMOVING PATTERNS CONTAINING NUMBERS AND CHARACTER COUNT < 3 --

# df['short_rr'] = df['short_rr'].apply(lambda x: ''.join(c for c in x if not c.isdigit()))
df = df[df['short_rr'] != '']

# -- COMMENT BELOW CODE TO CHECK FOR VALUES < 3 --

df['char_count'] = df['short_rr'].apply(lambda x: len(''.join(e for e in x if e.isalnum())))
df = df[df['char_count'] > 3]
# # REMOVING STOP WORDS
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
custom_stop_words = ['api', 'tcns', 'user', 'com', 'tcnsclothing', 'tcnssupport110', 'radiate', 'org', 
                     'tcnssupport1','tcnssupport2','tcnssupport3','tcnssupport4','tcnssupport5',
                     'tcnssupport8','tcnssupport7','tcnssupport6','tcnssupport9', 'seller', 'tcnssupportt3',
                     'technotask co','technotask', 'co', 'rto', 'dto', 'cocoblue', 'cx', 
                     'crmanish', 'tcnssupport', 'fmpr', 'myep', 'amz', 'pg', 'rvp', 'rtv', 'fmpp', 'dto',
                     'cr', 'app', 'channel', 'name', 'mesh', 'myntra','api', 'tcns', 'user', 'com', 'tcnsclothing', 'tcnssupport110', 'radiate', 'org', 
                     'tcnssupport1','tcnssupport2','tcnssupport3','tcnssupport4','tcnssupport5',
                     'tcnssupport8','tcnssupport7','tcnssupport6','tcnssupport9', 'seller', 'tcnssupportt3',
                     'technotask co','technotask', 'co', 'rto', 'dto', 'given','cocoblue', 'cx', 'customer', 
                     'crmanish', 'tcnssupport', 'fmpr', 'myep', 'amz', 'pg', 'rvp', 'rtv', 'fmpp', 'dto',
                     'cr', 'app', 'channel', 'name', 'mesh', 'flipkart', 'amazon', 'return',
                     'generic', 'claim','buyer', 'courier']

stop_words = set(stopwords.words('german'))
stop_words.update(custom_stop_words)

# -- EXCLUDING CERTAIN STOPWORDS --

#but added
word_to_exclude = "doesn"
stop_words = stop_words.difference({word_to_exclude})
word_to_exclude = "not"
stop_words = stop_words.difference({word_to_exclude})
word_to_exclude = "doesnt"
stop_words = stop_words.difference({word_to_exclude})
word_to_exclude = "no"
stop_words = stop_words.difference({word_to_exclude})
word_to_exclude = "dont"
stop_words = stop_words.difference({word_to_exclude})
word_to_exclude = "does"
stop_words = stop_words.difference({word_to_exclude})
word_to_exclude = "the"
stop_words = stop_words.difference({word_to_exclude})
word_to_exclude = "didnt"
stop_words = stop_words.difference({word_to_exclude})
word_to_exclude = "what"
stop_words = stop_words.difference({word_to_exclude})
word_to_exclude = "did"
stop_words = stop_words.difference({word_to_exclude})
word_to_exclude = "didn"
stop_words = stop_words.difference({word_to_exclude})
stop_words.update(stop_words)


final_reason = []

for string in df['short_rr']:
    words = word_tokenize(string)
    filtered_words = [word for word in words if word not in stop_words]
    filtered_string = ' '.join(filtered_words)
    final_reason.append(filtered_string)

    
df['short_rr'] = final_reason
df = df[['source_code', 'return_reason', 'short_rr' ,'rpi_count']]
df = df[df['short_rr'] != '']
# df['char_count'] = df['short_rr'].apply(lambda x: len(''.join(e for e in x if e.isalnum())))
# df = df[df['char_count'] > 3]
# ## WORD CORRECTION
corrected_words = ['misshipment', 'return', 'customer', 'delivered','panel', 'flipkart','mismatch', 
                   'different', 'wrong', 'comfort', 'level', 'received', 'confirmed', 'claim', 'missing', 'product', 
                   'quality', 'cancel', 'buyer', 'courier', 'defective', 'damaged','damage']


replaced_words = defaultdict(list)

def replace_words(text, corrected_words, threshold=0.8):
    corrected_string = []
    for word in text.split():
        max_similarity = max(textdistance.jaccard.normalized_similarity(word, cw) for cw in corrected_words)
        if max_similarity >= threshold:
            max_word = max(corrected_words, key=lambda cw: textdistance.jaccard.normalized_similarity(word, cw))
            if word != max_word:
                replaced_words[word].append(max_word)
            corrected_string.append(max_word)
        else:
            corrected_string.append(word)
    return ' '.join(corrected_string)

df['short_rr'] = df['short_rr'].apply(lambda x: replace_words(x, corrected_words, threshold=0.75))


# row removal x2
word_to_search = 'test'
df = df[~df['short_rr'].str.contains(word_to_search)]

word_to_search = 'myer'
df = df[~df['short_rr'].str.contains(word_to_search)]

word_to_search = 'crm'
df = df[~df['short_rr'].str.contains(word_to_search)]

word_to_search = 'inventory'
df = df[~df['short_rr'].str.contains(word_to_search)]

word_to_search = 'myec'
df = df[~df['short_rr'].str.contains(word_to_search)]

word_to_search = 'return expected on panel'
df = df[~df['short_rr'].str.contains(word_to_search)]

df = df[df['short_rr'] != 'undefined']
df = df[df['short_rr'] != 'rto'] 
df = df[df['short_rr'] != 'courier return'] 
df = df[df['short_rr'] != 'return'] 
df = df[df['short_rr'] != 'cr'] 
df = df[df['short_rr'] != 'customer return']
df = df[df['short_rr'] != 'rvp']
df = df[df['short_rr'] != 'rtv']
# pattern = r'rto\s+(?:' + '|'.join(return_reasons) + r')\b'
# df = df[~df['short_rr'].str.contains(pattern, regex=True)]
## space w/o space

pattern4 = 'ajio'
for text in df['short_rr']:
    matches = re.findall(pattern4, text)
df = df[~df['short_rr'].str.contains(pattern4)]

pattern4 = r'rto \d+$'
for text in df['short_rr']:
    matches = re.findall(pattern4, text)
matches = df[df['short_rr'].str.contains(pattern4)]['short_rr'].tolist()
df = df[~df['short_rr'].str.contains(pattern4)]

pattern4 = r'rto\d+$'
for text in df['short_rr']:
    matches = re.findall(pattern4, text)
matches = df[df['short_rr'].str.contains(pattern4)]['short_rr'].tolist()
df = df[~df['short_rr'].str.contains(pattern4)]

pattern4 = r'customer return \d+$'
for text in df['short_rr']:
    matches = re.findall(pattern4, text)
matches = df[df['short_rr'].str.contains(pattern4)]['short_rr'].tolist()
df = df[~df['short_rr'].str.contains(pattern4)]

pattern4 = r'customer return\d+$'
for text in df['short_rr']:
    matches = re.findall(pattern4, text)
matches = df[df['short_rr'].str.contains(pattern4)]['short_rr'].tolist()
df = df[~df['short_rr'].str.contains(pattern4)]

pattern4 = r'rtv \d+$'
for text in df['short_rr']:
    matches = re.findall(pattern4, text)
df = df[~df['short_rr'].str.contains(pattern4)]

pattern4 = r'rtv\d+$'
for text in df['short_rr']:
    matches = re.findall(pattern4, text)
df = df[~df['short_rr'].str.contains(pattern4)]

pattern4 = r'fmpc\d+$'
for text in df['short_rr']:
    matches = re.findall(pattern4, text)
df = df[~df['short_rr'].str.contains(pattern4)]

pattern4 = r'fmpc \d+$'
for text in df['short_rr']:
    matches = re.findall(pattern4, text)
df = df[~df['short_rr'].str.contains(pattern4)]

pattern4 = r'fmpr\d+$'
for text in df['short_rr']:
    matches = re.findall(pattern4, text)
df = df[~df['short_rr'].str.contains(pattern4)]

pattern4 = r'fmpr \d+$'
for text in df['short_rr']:
    matches = re.findall(pattern4, text)
df = df[~df['short_rr'].str.contains(pattern4)]

pattern4 = r'fmpp\d+$'
for text in df['short_rr']:
    matches = re.findall(pattern4, text)
df = df[~df['short_rr'].str.contains(pattern4)]

pattern4 = r'fmpp \d+$'
for text in df['short_rr']:
    matches = re.findall(pattern4, text)
df = df[~df['short_rr'].str.contains(pattern4)]
# word_to_search = "dto"
# df = df[~df['short_rr'].str.contains(word_to_search)]

## DTO stopword 'amz pg' , 'amz pg app'
# dto, rtv, fmpc, myec,

pattern4 = r'myep\d+$'
for text in df['short_rr']:
    matches = re.findall(pattern4, text)
df = df[~df['short_rr'].str.contains(pattern4)]

pattern4 = r'myep \d+$'
for text in df['short_rr']:
    matches = re.findall(pattern4, text)
df = df[~df['short_rr'].str.contains(pattern4)]

word_s = 'rtomanish'
df = df[~df['short_rr'].str.contains(word_s)]

word_s = ' ebo '
df = df[~df['short_rr'].str.contains(word_s)]

# word_s = ' rt '
# df = df[~df['short_rr'].str.contains(word_s)]

word_s = 'shipment bagout'
df = df[~df['short_rr'].str.contains(word_s)]

word_s = 'null null'
df = df[~df['short_rr'].str.contains(word_s)]

word_s = 'bag id'
df = df[~df['short_rr'].str.contains(word_s)]

df = df[~df['short_rr'].str.contains('origin')]

df = df[~df['short_rr'].str.contains('myn')]
df = df[~df['short_rr'].str.contains('limeroad')]
df = df[~df['short_rr'].str.contains('flex')]
df = df[~df['short_rr'].str.contains('nykaa')]
df = df[~df['short_rr'].str.contains('reason not available')]
df = df[~df['short_rr'].str.contains('others return reason')]
df = df[~df['short_rr'].str.contains(r'^received$')]
df = df[~df['short_rr'].str.contains('approved')]
df = df[~df['short_rr'].str.contains('pickup')]
df = df[~df['short_rr'].str.contains('address')]
df = df[~df['short_rr'].str.contains('manually')]
df = df[~df['short_rr'].str.contains('rto')]


# df['short_rr'] = df['short_rr'].apply(lambda x: x.replace("customer", ""))
df['short_rr'] = df['short_rr'].apply(lambda x: x.replace("recd", "received"))
df['short_rr'] = df['short_rr'].apply(lambda x: x.replace("damage", "damaged"))
df['short_rr'] = df['short_rr'].apply(lambda x: x.replace("damagedd", "damaged"))
df['short_rr'] = df['short_rr'].apply(lambda x: x.replace("colour", "color"))
df['short_rr'] = df['short_rr'].apply(lambda x: x.replace("issue", "issues"))
df['short_rr'] = df['short_rr'].apply(lambda x: x.replace("issuess", "issues"))
df['short_rr'] = df['short_rr'].apply(lambda x: x.replace("qc", "quality"))
df['short_rr'] = df['short_rr'].apply(lambda x: x.replace("undefined", ""))
df['short_rr'] = df['short_rr'].apply(lambda x: x.replace("cancelled", "cancel"))
df['short_rr'] = df['short_rr'].apply(lambda x: x.replace("use", "used"))
df['short_rr'] = df['short_rr'].apply(lambda x: x.replace("tag ", "tags "))
df['short_rr'] = df['short_rr'].apply(lambda x: x.replace("defective", "damaged"))
df['short_rr'] = df['short_rr'].apply(lambda x: x.replace("item", "product"))

pattern = r'^[0-9-]+$'
df = df[~df['short_rr'].str.contains(pattern)]
df['short_rr'] = df['short_rr'].apply(lambda x: ''.join(c for c in x if not c.isdigit()))
df = df[df['short_rr'] != '']
df['char_count'] = df['short_rr'].apply(lambda x: len(''.join(e for e in x if e.isalnum())))
df = df[df['char_count'] > 3]


def remove_extra_characters(word):
    return re.sub(r'(.)\1+', r'\1\1', word)
df['short_rr'] = df['short_rr'].apply(remove_extra_characters)

grouped = df.groupby(by='short_rr')['rpi_count'].sum().sort_values(ascending=False)
grouped = pd.DataFrame(grouped)
grouped = grouped[grouped > 10]
grouped.dropna(inplace=True)
grouped.reset_index(inplace=True)
# df.groupby(by='short_rr')['rpi_count'].sum().sort_values(ascending=False).to_excel("/Users/anubhavgupta/Desktop/Return_Reasons_Project/Excels/Cleaned Data/test_data_after_algo.xlsx")
# grouped.to_excel("/Users/anubhavgupta/Desktop/Return_Reasons_Project/Excels/Final Algorithm/Unclassified Data/2022.xlsx")

# df = pd.read_excel("/Users/anubhavgupta/Desktop/Return_Reasons_Project/Excels/Final Algorithm/Unclassified Data/2022.xlsx")
subclasses = ['fit issue',
 'small size',
 'large size',
 'did not like product',
 'product image better',
 'not required anymore',
 'wrong product recevied',
 'damaged product',
 'material issues',
 'delivery issue', 
 'product missing',
 'ordered incorrectly',
 'found better price',
 'price related',
 'unsatisfactory product']



candidate_labels = subclasses
def classify_return_reasons(df, classifier, candidate_labels):
    classifications = []
    for return_reason in df['short_rr']:
        classification = classifier(return_reason, candidate_labels, multi_label=False)
        highest_label = max(classification['scores'])
        highest_label_idx = classification['scores'].index(highest_label)
        highest_label_name = classification['labels'][highest_label_idx]
        classifications.append(highest_label_name)
    df['classification'] = classifications
    return df

df = classify_return_reasons(df, classifier, candidate_labels)
# (classified_df[['short_rr', 'classification','rpi_count']])
df.to_csv("/tmp/categorised.csv")
# df.to_excel('/Users/anubhavgupta/Desktop/Return_Reasons_Project/Excels/Final Algorithm/Classified Data/2022.xlsx')
