# Changelog

## [0.2.0](https://github.com/gr8monk3ys/huggingface/compare/v0.1.0...v0.2.0) (2026-09-11)


### Features

* **dataset-explorer:** rebuild as a static Space, and fix every broken preset ([#31](https://github.com/gr8monk3ys/huggingface/issues/31)) ([dc8e8db](https://github.com/gr8monk3ys/huggingface/commit/dc8e8dbf4693d1f405859db778d3b81a1bcb2bd8))

## 0.1.0 (2026-09-09)


### Features

* **model-selector:** query the Hub live with a curated fallback ([823a87d](https://github.com/gr8monk3ys/huggingface/commit/823a87df6e5947b695210be8e1bc716d9da64803))
* **model-selector:** rebuild as a static Space so it is always awake ([#28](https://github.com/gr8monk3ys/huggingface/issues/28)) ([537fbda](https://github.com/gr8monk3ys/huggingface/commit/537fbdaf46a0b36e6441bfb38190e44415003209))
* one-shot Hub publish script ([c523fe3](https://github.com/gr8monk3ys/huggingface/commit/c523fe3af144d936a00432b0efd9da6557df8f12))
* **paper-recommender:** load the real arXiv dataset with a fallback ([87df6c5](https://github.com/gr8monk3ys/huggingface/commit/87df6c55806a60a22e7a0ae33bb9f7a8ae1897fc))
* rebuild ml-interview as a static Space so it is always on ([cf667f1](https://github.com/gr8monk3ys/huggingface/commit/cf667f1d300dba19a40b8439b59655cd7be36574))
* **research-assistant:** agentic RAG Space over the arXiv papers ([45226df](https://github.com/gr8monk3ys/huggingface/commit/45226dfc43e208beedbceaf18fb086ce418a7556))


### Bug Fixes

* **ci:** repoint org workflows to the public reusable home ([#11](https://github.com/gr8monk3ys/huggingface/issues/11)) ([a70e6c1](https://github.com/gr8monk3ys/huggingface/commit/a70e6c184cf464360976ecb0a552570411f75d98))
* clear the three semgrep findings failing the monthly scan ([93b11e6](https://github.com/gr8monk3ys/huggingface/commit/93b11e6bc26a3f736393eeeb0c43ca11936db125))
* create_repo(exist_ok=True) aborted the run on existing Spaces ([123a407](https://github.com/gr8monk3ys/huggingface/commit/123a4075a72eaa362357c024d29136cfaafe65b7))
* don't let a model download failure cost the Space its UI ([edf94cf](https://github.com/gr8monk3ys/huggingface/commit/edf94cf0235694e8d702d5e7205a24f776d1fe0b))
* make inference features work in production + fix ML correctness bugs ([4698280](https://github.com/gr8monk3ys/huggingface/commit/469828002ac4d9824d88c67e52ff8a47d4414c51))
* migrate training to transformers v5, clearing the RCE advisory ([60a308b](https://github.com/gr8monk3ys/huggingface/commit/60a308b1dc143b522b4daa3140e406ac62f5386a))
* paper-classifier card pointed at a Hub repo that does not exist ([159fc4f](https://github.com/gr8monk3ys/huggingface/commit/159fc4f4da097d337059495f5412a12ea809a861))
* place nosemgrep on the line semgrep actually checks ([5b24086](https://github.com/gr8monk3ys/huggingface/commit/5b2408682de7a8e34f160af9d4dcd98968b0ce4a))
* **publish:** stop uploading arrow files that disable the dataset viewer ([#30](https://github.com/gr8monk3ys/huggingface/issues/30)) ([d330112](https://github.com/gr8monk3ys/huggingface/commit/d330112a3d998b8bc8579043ba05975571f4db1b))
* report exhausted inference credits as billing, not a bad token ([96bf6af](https://github.com/gr8monk3ys/huggingface/commit/96bf6af3565b9bdc3de60f841bb8832ba0b9a8ed))
* **research-assistant:** Mistral v0.3 is retired, use v0.2 ([#26](https://github.com/gr8monk3ys/huggingface/issues/26)) ([3bf9d1a](https://github.com/gr8monk3ys/huggingface/commit/3bf9d1aa2897fb6627ffff2bd12bd1d41d5fffeb))
* stratified split crashed -- cast label to ClassLabel first ([fa559d8](https://github.com/gr8monk3ys/huggingface/commit/fa559d80077c341628d7430658a0b963e3bd22fe))
* **training:** converge the two pipelines, and make them runnable again ([#25](https://github.com/gr8monk3ys/huggingface/issues/25)) ([c1430e3](https://github.com/gr8monk3ys/huggingface/commit/c1430e3d0f7b7eae38d3da2f379229b2003ea9f6))
* update gradio to &gt;=5.31.0 across all spaces to resolve critical ACL bypass and DoS vulnerabilities ([a55e7f5](https://github.com/gr8monk3ys/huggingface/commit/a55e7f523307fc50f64e4ec36e26a51a897dcc1f))


### Documentation

* correct the project count and document the missing workflows ([65b2411](https://github.com/gr8monk3ys/huggingface/commit/65b24115800c115d3262339800b3177911458c08))
* document all 15 projects, reconcile versions, de-hype claims ([42be148](https://github.com/gr8monk3ys/huggingface/commit/42be1485e4e1ddfe9f5f6591b2231cba190a62dd))
* give every Hub card the metadata search actually uses ([#27](https://github.com/gr8monk3ys/huggingface/issues/27)) ([b7c0749](https://github.com/gr8monk3ys/huggingface/commit/b7c0749d7e671a0b20f13170ea86df39e2a00c1b))
* record measured results in place of estimates ([2fff69e](https://github.com/gr8monk3ys/huggingface/commit/2fff69ef0d801c6029a86002bb155f6fc02804a8))
* record the three decisions the refactor sequence is written against ([#16](https://github.com/gr8monk3ys/huggingface/issues/16)) ([c184208](https://github.com/gr8monk3ys/huggingface/commit/c1842083fb9427682c6d6cb00a64cf08a14cab15))
* write up the summarizer outage ([#29](https://github.com/gr8monk3ys/huggingface/issues/29)) ([f34a013](https://github.com/gr8monk3ys/huggingface/commit/f34a01343edff344939ae1c82803f27a05cd3460))
