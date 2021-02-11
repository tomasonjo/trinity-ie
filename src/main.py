import itertools
import json
import urllib
from string import punctuation

import nltk
import spacy
from flask import Flask, request

import neuralcoref
import opennre

ENTITY_TYPES = ["human", "person", "company", "enterprise", "business", "geographic region",
                "human settlement", "geographic entity", "territorial entity type", "organization"]

# Load SpaCy
nlp = spacy.load('en')
# Add neural coref to SpaCy's pipe
neuralcoref.add_to_pipe(nlp)

# Load opennre
relation_model = opennre.get_model('wiki80_bert_softmax')

# Load NLTK
nltk.download('punkt')


def wikifier(text, lang="en", threshold=0.8):
    """Function that fetches entity linking results from wikifier.com API"""
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
        # Filter out desired entity classes
        if ('wikiDataClasses' in annotation) and (any([el['enLabel'] in ENTITY_TYPES for el in annotation['wikiDataClasses']])):

            # Specify entity label
            if any([el['enLabel'] in ["human", "person"] for el in annotation['wikiDataClasses']]):
                label = 'Person'
            elif any([el['enLabel'] in ["company", "enterprise", "business", "organization"] for el in annotation['wikiDataClasses']]):
                label = 'Organization'
            elif any([el['enLabel'] in ["geographic region", "human settlement", "geographic entity", "territorial entity type"] for el in annotation['wikiDataClasses']]):
                label = 'Location'
            else:
                label = None

            results.append({'title': annotation['title'], 'wikiId': annotation['wikiDataItemId'], 'label': label,
                            'characters': [(el['chFrom'], el['chTo']) for el in annotation['support']]})
    return results


def coref_resolution(text):
    """Function that executes coreference resolution on a given text"""
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
    """Removes all punctuation from a string"""
    return ''.join(c for c in s if c not in punctuation)


def deduplicate_dict(d):
    return [dict(y) for y in set(tuple(x.items()) for x in d)]


app = Flask(__name__)


@ app.route('/')
def hello_ie():
    try:
        text = request.args.get('text', None)
        relation_threshold = request.args.get('relation_threshold', 0.9)
        entities_threshold = request.args.get('entities_threshold', 0.8)
        coref = request.args.get('coref', True)
        if not text:
            return 'Missing text parameter'

        try:
            relation_threshold = float(relation_threshold)
            entities_threshold = float(entities_threshold)
        except ValueError:
            return 'Invalid value for relation or entity threshold parameter'

        if coref:
            text = coref_resolution(text)

        print(text)

        relations_list = list()
        entities_list = list()

        for sentence in nltk.sent_tokenize(text):
            sentence = strip_punctuation(sentence)
            entities = wikifier(sentence, threshold=entities_threshold)
            entities_list.extend(
                [{'title': el['title'], 'wikiId': el['wikiId'], 'label': el['label']} for el in entities])
            # Iterate over every permutation pair of entities
            for permutation in itertools.permutations(entities, 2):
                for source in permutation[0]['characters']:
                    for target in permutation[1]['characters']:
                        # Relationship extraction with OpenNRE
                        data = relation_model.infer(
                            {'text': sentence, 'h': {'pos': [source[0], source[1] + 1]}, 't': {'pos': [target[0], target[1] + 1]}})
                        if data[1] > relation_threshold:
                            relations_list.append(
                                {'source': permutation[0]['title'], 'target': permutation[1]['title'], 'type': data[0]})

        return {'entities': deduplicate_dict(entities_list), 'relations': deduplicate_dict(relations_list)}
    except Exception as e:
        return 'An error has occured:' + str(e)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
