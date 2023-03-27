import emoji
import re
import torch

def normalize_token(token):
    if len(token) == 1:
        return emoji.demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == '…':
            return "..."
        else:
            return token

def isnan(s):
    return s != s


def normalize_post(post, tweet_tokenizer, vncorenlp, use_segment=False, remove_punc_stopword=False):
    tokens = tweet_tokenizer.tokenize(post.replace("’", "'").replace("…", "..."))
    post = " ".join(tokens)
    
    if use_segment:
        tokens = vncorenlp.tokenize(post.replace("’", "'").replace("…", "..."))
        tokens = [t for ts in tokens for t in ts]
    
    norm_post = " ".join(tokens)
    
    norm_post = re.sub(r",([0-9]{2,4}) , ([0-9]{2,4})", r",\1,\2", norm_post)
    norm_post = re.sub(r"([0-9]{1,3}) / ([0-9]{2,4})", r"\1/\2", norm_post)
    norm_post = re.sub(r"([0-9]{1,3})- ([0-9]{2,4})", r"\1-\2", norm_post)
    
    if use_segment:
        norm_post = norm_post.replace('< url >', '<url>')
        norm_post = re.sub(r"# (\w+)", r'#\1', norm_post)
    
    return norm_post

def convert_samples_to_ids(texts, tokenizer, max_seq_length=256, labels=None):
    input_ids, attention_masks = [], []
    
    for text in texts:
        inputs = tokenizer.enocde_plus(text, padding='max_length', max_seq_length=max_seq_length, truncation=True)
        input_ids.append(inputs['input_ids'])
        attention_masks.append(inputs['attention_mask'])
        
    if labels is not None:
        return torch.tensor(input_ids, dtype=torch.long), \
                torch.tensor(attention_masks, dtype=torch.long), \
                torch.tensor(labels, dtype=torch.long)
    
def get_max_seq(texts, tokenizer):
    max_seq_length = []
    for text in texts:
        tokens = tokenizer.tokenize(text)
        max_seq_length.append(len(tokens))
    
    return max_seq_length