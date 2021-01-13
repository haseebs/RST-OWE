# Relation Specific Transformations for Open World Knowledge Graph Completion
This repository contains the official Pytorch code for [Relation Specific Transformations for Open World Knowledge Graph Completion](https://www.aclweb.org/anthology/2020.textgraphs-1.9). This code is built on top of our original OWE codebase.

## Setup

Resolve dependencies by executing the following command:
```bash
pip install -e .
```

Then download the
[FB15k-237-OWE](http://haseeb.ai/assets/FB15k-237-OWE.zip) and
[DBPedia50](http://haseeb.ai/assets/dbpedia50.zip) datasets as
required. These contain the datasets in format required for both OpenKE
and our code.

## Usage
### Training the KGC model

Before using OWE you need to train a KGC model. You can use any knowledge graph embedding framework for training. 
We used the PyTorch version of [OpenKE](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch).  

To use a previously trained KGC model with OWE you need to export the entity and relation matrices as 
*numpy arrays* using pickle:
 - For TransE and DistMult the matrices are called `entities.p` and `relations.p`
 - For CompLEx `entities_r.p`, `entities_i.p`, `relations_r.p` and `relations_i.p`.  

Furthermore you need to provide two more files: `entity2id.txt` and `relation2id.txt`, which contain a mapping
of an entity/relation string to the corresponding index (id) in the embedding files.

The trained 300D ComplEx embeddings for FB15k-237-OWE can be obtained from [here](http://haseeb.ai/assets/FB15k-237-OWE-closed-world-embeddings.zip).

For closed-world evaluation:
```bash
python owe/run_closed_world.py -e -c <PATH_TO_DATASET> --<KGC_MODEL_NAME> <PATH_TO_KGC_EMBEDDINGS>
```
where `KGC_MODEL_NAME` can be `complex`, `distmult` or `transe`.

#### Instructions for OpenKE

Follow the instructions to install OpenKE. Under [scripts](scripts/openke_training) you can find exemplary training scripts (make sure you update the paths to the dataset in OpenKE format). As the API of OpenKE frequently changes, scripts work with the 
[following version of OpenKE](https://github.com/thunlp/OpenKE/tree/0a55399b3e800bc779582c4784cac96f00230fd8).

### Training the Open-World Extention

An example config file is provided as config.ini in the root folder.

For training:
```bash
python owe/main.py -t -c <PATH_TO_DATASET> -d  <PATH_TO_CONFIG_AND_OUTPUT_DIR> --<KGC_MODEL_NAME> <PATH_TO_KGC_EMBEDDINGS>
```

For evaluation:
```bash
python owe/main.py -e -lb -c <PATH_TO_DATASET> -d  <PATH_TO_CONFIG_AND_OUTPUT_DIR> --<KGC_MODEL_NAME> <PATH_TO_KGC_EMBEDDINGS>
```

#### Reproducing the paper experiments
The configs used to generated the experiments in the paper alongwith
their respective output log files are provided in the
`paper_cfgs_and_logs` folder. We recommend using those as a starting
point.

## Citation
If you found our work helpful in your research, consider citing the following:
```
@inproceedings{shah-etal-2020-relation,
    title = "Relation Specific Transformations for Open World Knowledge Graph Completion",
    author = "Shah, Haseeb and Villmow, Johannes and Ulges, Adrian",
    booktitle = "Proceedings of the Graph-based Methods for Natural Language Processing (TextGraphs)",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.textgraphs-1.9",
    pages = "79--84",
}
```
