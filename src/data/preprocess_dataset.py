from .graph_creation import Dataset
from .preprocessing import NER_stanza
from .preprocessing import SRL
import torch


def add_metadata2graph(graph, metadata):
    for (node, dict_node) in metadata.items():
        for (k, v) in dict_node.items():
            if node in graph.ntypes:
                graph.nodes[node].data[k] = torch.tensor(v)
    return graph


def create_dataloader(hotpot, dict_ins2dict_doc2pred, pretrained_weights):
    # extract entities and SRL
    ner = NER_stanza()
    srl = SRL()
    print("Extracting named entities from the query")
    list_ent_query = ner.extract_named_entities_from_query(hotpot)
    print("Extracting named entities")
    list_hotpot_ner = ner.extract_named_entities(hotpot, dict_ins2dict_doc2pred)
    print("Extracting SRL arguments from the query")
    dict_ins_query_srl_triples = srl.extract_srl_from_query(hotpot)
    print("Extracting SRL arguments")
    dict_ins_doc_sent_srl_triples = srl.extract_srl(hotpot, dict_ins2dict_doc2pred)

    print("Data loaded. Creating graphs")
    train_dataset = Dataset(hotpot, list_hotpot_ner, dict_ins_doc_sent_srl_triples,
                            dict_ins_query_srl_triples, list_ent_query, 
                            dict_ins2dict_doc2pred=dict_ins2dict_doc2pred, batch_size=1,
                            pretrained_weights=pretrained_weights)
    (list_graphs,
        list_context,
        list_span_idx) = train_dataset.create_dataloader()

    return {'list_graphs': list_graphs,
            'list_context': list_context,
            'list_span_idx': list_span_idx}


#create_dataloader('/workspace/ml-workspace/thesis_git/HSGN/data/')
