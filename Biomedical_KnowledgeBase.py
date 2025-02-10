import json
import stanza
import warnings
from pymongo import MongoClient
from openie import StanfordOpenIE
import xml.etree.ElementTree as ET

warnings.filterwarnings("ignore")

# Download necessary Stanza models
biomedical_models = [
    'bc5cdr', 'bionlp13cg', 'jnlpba', 'linnaeus',
    'ncbi_disease', 's800', 'i2b2', 'radiology'
]

# Load each biomedical model into its own pipeline
pipelines = {}
for model in biomedical_models:
    stanza.download('en', package=model)
    pipelines[model] = stanza.Pipeline(
        lang='en',
        processors={'tokenize': 'default', 'ner': model},
        download_method=None
    )

with StanfordOpenIE() as openie:
    def process_sentence(sentence):
        tokens = []
        pos_tags = []
        lemmas = []
        entities = []

        # Run each pipeline and aggregate results
        for model_name, pipeline in pipelines.items():
            stanza_doc = pipeline(sentence)

            tokens.extend([word.text for word in stanza_doc.sentences[0].words])
            pos_tags.extend([(word.text, word.xpos) for word in stanza_doc.sentences[0].words])
            lemmas.extend([word.lemma for word in stanza_doc.sentences[0].words])
            entities.extend([(ent.text, ent.type) for ent in stanza_doc.ents])

        # OpenIE for extracting triples
        sentence_triples = openie.annotate(sentence)

        return {
            "sentence": sentence,
            "tokens": tokens,
            "pos_tags": pos_tags,
            "entities": entities,
            "lemmas": lemmas,
            "triples": sentence_triples
        }


    def process_text(text):
        # Sentence splitting using the first pipeline
        stanza_doc = list(pipelines.values())[0](text)
        sentences = [sentence.text for sentence in stanza_doc.sentences]

        # Process each sentence individually
        sentence_data = []
        for sentence in sentences:
            sentence_info = process_sentence(sentence)
            sentence_data.append(sentence_info)

        return sentence_data


    # Create knowledge base list
    knowledge_base = []

    # Parse the XML file
    file_path = 'pubmed24n1220.xml'
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Iterating over each article in the XML file and extract data
    for article in root.findall('.//PubmedArticle'):
        pmid_element = article.find('.//PMID')
        title_element = article.find('.//ArticleTitle')
        abstract_element = article.find('.//Abstract/AbstractText')

        pmid = pmid_element.text if pmid_element is not None else None
        title = title_element.text if title_element is not None else "No Title"
        abstract = abstract_element.text if abstract_element is not None else None

        # Skip if the abstract is missing
        if not abstract:
            continue

        # Process each abstract text in the XML file
        sentence_data = process_text(abstract)

        # Create JSON object for the knowledge base
        knowledge_base_entry = {
            "PMID": pmid,
            "title": title,
            "abstract": abstract,
            "sentences": sentence_data
        }
        knowledge_base.append(knowledge_base_entry)

        # Save the knowledge base to a JSON file
    with open("biomedical_knowledge_base.json", "w") as f:
        json.dump(knowledge_base, f, indent=4)
        print("Knowledge base saved to JSON file.")

        # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["BIOMEDICAL_ARTICLES"]
    collection = db["MedicalKnowledgeBase"]

    # Load data from JSON file and insert into MongoDB
    with open("biomedical_knowledge_base.json", "r") as f:
        knowledge_base_data = json.load(f)
        if knowledge_base_data:
            collection.insert_many(knowledge_base_data)
            print("All documents processed and stored in MongoDB.")
