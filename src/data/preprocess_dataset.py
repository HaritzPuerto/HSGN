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

def create_dataloader(hotpot):
    # extract entities and SRL
    ner = NER_stanza()
    srl = SRL()
    print("Extracting named entities from the query")
    list_ent_query_training = ner.extract_named_entities_from_query(hotpot)
    print("Extracting named entities")
    list_hotpot_train_ner = ner.extract_named_entities(hotpot)
    print("Extracting SRL arguments from the query")
    dict_ins_query_srl_triples = srl.extract_srl_from_query(hotpot)
    print("Extracting SRL arguments")
    dict_ins_doc_sent_srl_triples = srl.extract_srl(hotpot)

    print("Data loaded. Creating graphs")
    train_dataset = Dataset(hotpot, list_hotpot_train_ner, dict_ins_doc_sent_srl_triples,
                            dict_ins_query_srl_triples, list_ent_query_training, batch_size=1)
    (list_graphs,
        list_g_metadata,
        list_context,
        list_list_srl_edges_metadata,
        list_list_ent2ent_metadata,
        list_span_idx) = train_dataset.create_dataloader()

    for g_idx, list_dict_edge in enumerate(list_list_ent2ent_metadata):
        if 'ent2ent_rel' in list_graphs[g_idx].etypes:
            list_graphs[g_idx].edges['ent2ent_rel'].data['rel_type'] = torch.tensor([edge['rel_type'] for edge in list_dict_edge])
            list_graphs[g_idx].edges['ent2ent_rel'].data['span_idx'] = torch.tensor([edge['span_idx'] for edge in list_dict_edge])

    list_graph_metadata = list(zip(list_graphs, list_g_metadata))
    list_graphs2 = []
    for (g, metadata) in list_graph_metadata:
        # add metadata to the graph
        graph = add_metadata2graph(g, metadata)
        list_graphs2.append(graph)

    return {'list_graphs': list_graphs2,
            'list_context': list_context,
            'list_span_idx': list_span_idx}


#create_dataloader('/workspace/ml-workspace/thesis_git/HSGN/data/')
