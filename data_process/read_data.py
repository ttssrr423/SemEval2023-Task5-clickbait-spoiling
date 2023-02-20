import pandas as pd

def input_data_to_squad_format(inp):
    # return pd.DataFrame([{'id': i['uuid'], 'title': i['targetTitle'], 'question': ' '.join(i['postText']), 'context': i['targetTitle'] + ' - ' + (' '.join(i['targetParagraphs'])), 'answers': 'not available for predictions'} for i in inp])
    return pd.DataFrame([{'id': i['uuid'], 'title': i['targetTitle'], 'question': ' '.join(i['postText']),
                          'context': i['targetTitle'] + ' - ' + (' '.join(i['targetParagraphs'])),
                          'answers': " ".join(i["spoiler"])} for i in inp])