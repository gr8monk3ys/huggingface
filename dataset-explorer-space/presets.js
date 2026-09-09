// Quick-start datasets for the explorer.
//
// Every id here was verified against the datasets-server before being listed.
// The previous list used legacy short names -- "imdb", "squad", "glue" -- and
// all ten had been renamed to namespaced ids, so every shortcut button in the
// Space returned "The dataset has been renamed" and nothing worked.
//
// tests/test_dataset_explorer_data.py parses this file and checks the shape.
// It cannot check that the ids still resolve, because tests here are offline;
// if a button starts failing, this is the list to re-verify.
const PRESETS = [
  { id: "stanfordnlp/imdb", label: "IMDB reviews" },
  { id: "rajpurkar/squad", label: "SQuAD" },
  { id: "nyu-mll/glue", label: "GLUE" },
  { id: "Salesforce/wikitext", label: "WikiText" },
  { id: "fancyzhx/ag_news", label: "AG News" },
  { id: "Yelp/yelp_review_full", label: "Yelp reviews" },
  { id: "fancyzhx/amazon_polarity", label: "Amazon polarity" },
  { id: "fancyzhx/dbpedia_14", label: "DBpedia 14" },
  { id: "dair-ai/emotion", label: "Emotion" },
  { id: "gr8monk3ys/academic-papers-dataset", label: "arXiv CS/ML papers" },
];
