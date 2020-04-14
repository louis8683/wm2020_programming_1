NUMBER = 0
TITLE = 1
CONCEPTS = 2


def read_queries(filename):
        import xml.etree.ElementTree as ET
        tree = ET.parse(filename)
        root = tree.getroot()
        queries = []
        for topic in root:
            # Every topic
            number, title, question, narrative, concepts = topic
            query_id = number.text[-3:]
            title = title.text
            concepts = concepts.text.split("\n")[1].split("。")[0].split("、")
            queries.append((query_id, title, concepts))
        return queries
