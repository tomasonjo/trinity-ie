import itertools
import json
import urllib
from string import punctuation

import nltk
import spacy
from flask import Flask, request

import neuralcoref
import opennre

# Load SpaCy
nlp = spacy.load('en')
# Add neural coref to SpaCy's pipe
neuralcoref.add_to_pipe(nlp)

# Load opennre
relation_model = opennre.get_model('wiki80_cnn_softmax')

# Load NLTK
nltk.download('punkt')


def wikifier(text, lang="en", threshold=0.8):
    # Prepare the URL.
    data = urllib.parse.urlencode([
        ("text", text), ("lang", lang),
        ("userKey", "tgbdmkpmkluegqfbawcwjywieevmza"),
        ("pageRankSqThreshold", "%g" %
         threshold), ("applyPageRankSqThreshold", "true"),
        ("nTopDfValuesToIgnore", "100"), ("nWordsToIgnoreFromList", "100"),
        ("wikiDataClasses", "true"), ("wikiDataClassIds", "false"),
        ("support", "true"), ("ranges", "false"), ("minLinkFrequency", "2"),
        ("includeCosines", "false"), ("maxMentionEntropy", "3")
    ])
    url = "http://www.wikifier.org/annotate-article"
    # Call the Wikifier and read the response.
    req = urllib.request.Request(url, data=data.encode("utf8"), method="POST")
    with urllib.request.urlopen(req, timeout=60) as f:
        response = f.read()
        response = json.loads(response.decode("utf8"))
    # Output the annotations.
    results = list()
    for annotation in response["annotations"]:
        results.append({'title': annotation['title'], 'wikiId': annotation['wikiDataItemId'],
                        'characters': [(el['chFrom'], el['chTo']) for el in annotation['support']]})

    return results


def coref_resolution(text):
    doc = nlp(text)
    # fetches tokens with whitespaces from spacy document
    tok_list = list(token.text_with_ws for token in doc)
    for cluster in doc._.coref_clusters:
        # get tokens from representative cluster name
        cluster_main_words = set(cluster.main.text.split(' '))
        for coref in cluster:
            if coref != cluster.main:  # if coreference element is not the representative element of that cluster
                if coref.text != cluster.main.text and bool(set(coref.text.split(' ')).intersection(cluster_main_words)) == False:
                    # if coreference element text and representative element text are not equal and none of the coreference element words are in representative element. This was done to handle nested coreference scenarios
                    tok_list[coref.start] = cluster.main.text + \
                        doc[coref.end-1].whitespace_
                    for i in range(coref.start+1, coref.end):
                        tok_list[i] = ""

    return "".join(tok_list)

def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)


app = Flask(__name__)


@app.route('/')
def hello_ie():
    text = request.args.get('text', None)
    relation_threshold = request.args.get('relation_threshold', 0.2)
    entities_threshold = request.args.get('entities_threshold', 0.9)
    coref = request.args.get('coref', True)

    if not text:
        return 'Missing text parameter'

    if coref:
        text = coref_resolution(text)

    print(text)

    relations_list = list()
    entities_set = set()

    for sentence in nltk.sent_tokenize(text):
        sentence = strip_punctuation(sentence)
        entities = wikifier(sentence, threshold=entities_threshold)
        entities_set.update([(el['title'], el['wikiId']) for el in entities])
        for permutation in itertools.permutations(entities, 2):
            for source in permutation[0]['characters']:
                for target in permutation[1]['characters']:
                    data = relation_model.infer(
                        {'text': sentence, 'h': {'pos': source}, 't': {'pos': target}})
                    if data[1] > relation_threshold:
                        relations_list.append(
                            {'source': permutation[0]['wikiId'], 'target': permutation[1]['wikiId'], 'type': data[0]})

    return {'entities': list(entities_set), 'relations': [dict(y) for y in set(tuple(x.items()) for x in relations_list)]}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
