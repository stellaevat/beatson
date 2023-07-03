from collections import defaultdict

wordlists = ["terms_cancers.tsv", "terms_conflicting.tsv", "terms_drugs.tsv", "terms_genes.tsv", "terms_proteins.tsv", "terms_variants.tsv",]

def get_terms_and_synonyms(wordlist):
  terms = set()
  with open(wordlist, encoding="utf8") as f:
    synonyms = [line.strip().split("\t")[2] for line in f]
    for group in synonyms:
      terms.update(set(group.split("|")))
  return terms, synonyms

def get_vocabulary_and_index(terms):
  vocabulary = {}
  index = {}
  counter = 0
  for term in terms:
      vocabulary[term] = counter
      index[counter] = term
      counter += 1
  return vocabulary, index

def get_synonym_graph(wordlists=wordlists):
  all_terms = set()
  all_synonyms = []
  synonym_graph = defaultdict(list)

  for wordlist in wordlists:
    terms, synonyms = get_terms_and_synonyms("wordlists/" + wordlist)
    all_terms.update(terms)
    all_synonyms.extend(synonyms)

  vocabulary, index = get_vocabulary_and_index(all_terms)

  for group_str in all_synonyms:
    group = group_str.split("|")
    for target in group:
      for source in group:
        if target != source:
          # TODO: If target contains source maybe don't need to include
          synonym_graph[vocabulary[source]].append(vocabulary[target])
  
  return synonym_graph, vocabulary, index
