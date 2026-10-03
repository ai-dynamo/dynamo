# Bibliography: learned-routing campaign literature sweep (2026-10-02)

This is the master list of sources the four scouts found. The synthesized lessons are in
`<campaign-root>/literature/LESSONS.md`; IDs LR-01 to LR-15 refer to that
file.

**Where things are**

| What | Path |
|---|---|
| PDFs | `/tmp/learned-routing-lit/pdfs/<area>/` |
| Text extractions | `/tmp/learned-routing-lit/txt/<area>/` |
| Scout notes | `/tmp/learned-routing-lit/notes/{dcm-context,llm-routing,learned-systems-policies,eval-methodology}.md` |

**Read depth**

| Value | Meaning |
|---|---|
| full | The whole paper was read. |
| key | The key sections were read. |
| abstract | Only the abstract was read. |
| web | Read online; not downloaded. |

## Discrete choice, context effects, choice models as policies (`pdfs/dcm-context/`, notes `dcm-context.md`)

| Citation | URL | Local file | Depth | Used in |
|---|---|---|---|---|
| Tomlinson & Benson 2021. Learning Interpretable Feature Context Effects in Discrete Choice. KDD | https://arxiv.org/abs/2009.03417 | `tomlinson2021-lcl-dlcl.pdf` | full | LR-05, LR-06, LR-15 |
| Seshadri, Peysakhovich & Ugander 2019. Discovering Context Effects from Raw Choice Data. ICML | https://arxiv.org/abs/1902.03266 | `seshadri2019-cdm.pdf` | full | LR-05, LR-15 |
| Rosenfeld, Oshiba & Singer 2020. Predicting Choice with Set-Dependent Aggregation. ICML | https://arxiv.org/abs/1906.06365 | `rosenfeld2020-set-dependent-aggregation.pdf` | key | LR-05, LR-06 |
| Pfannschmidt, Gupta, Haddenhorst & Hüllermeier 2022. Learning Context-Dependent Choice Functions. IJAR | https://arxiv.org/abs/1901.10860 | `pfannschmidt2019-fate-feta.pdf` | key | LR-06 |
| Aouad & Désir 2023. Representing Random Utility Choice Models with Neural Networks (RUMnet). Management Science | https://arxiv.org/abs/2207.12877 | `aouad2023-rumnet.pdf` | key | LR-15 |
| Yousefi Maragheh, Chronopoulou & Davis 2018. A Customer Choice Model with HALO Effect. arXiv | https://arxiv.org/abs/1805.01603 | `maragheh2018-halo-mnl.pdf` | key | design confirmation (no worker-ID terms) |
| Ko & Li 2024. Modeling Choice via Self-Attention. arXiv | https://arxiv.org/abs/2311.07607 | `ko2024-choice-self-attention.pdf` | key | LR-05 |
| Zhang, Wang, Gao & Li 2026. DeepHalo: A Neural Choice Model with Controllable Context Effects. arXiv | https://arxiv.org/abs/2601.04616 | `zhang2026-deephalo.pdf` | key | LR-15 |
| Train 2009. Discrete Choice Methods with Simulation, 2nd ed., Ch. 4 (GEV). Cambridge UP | https://eml.berkeley.edu/books/choice2nd/Ch04_p76-96.pdf | `train2009-gev-ch4.clean.pdf`; the original `train2009-gev-ch4.pdf` is MacBinary-wrapped | key | LR-15 |
| Seshadri & Ugander 2020. Fundamental Limits of Testing the IIA in Discrete Choice. EC 2019, extended | https://arxiv.org/abs/2001.07042 | `seshadri2020-iia-testing-limits.pdf` | abstract | LR-11 |
| Webb, Glimcher & Louie 2021. The Normalization of Consumer Valuations. Management Science | https://www.neuroeconomicslab.org/s/Normalization-of-Consumer-Valuations.pdf | `webb2021-divisive-normalization.pdf` | key | LR-06, LR-15 |
| Jain, Szot & Lim 2020. Generalization to New Actions in Reinforcement Learning. ICML | https://arxiv.org/abs/2011.01928 | `jain2020-generalization-new-actions.pdf` | key | LR-10 |
| Jain, Kosaka, Kim & Lim 2022. Know Your Action Set: Learning Action Relations for RL (AGILE). ICLR | https://openreview.net/forum?id=MljXVdp4A3N | `jain2022-agile-action-relations.pdf` (ICLR slides, not the paper) | abstract | background |
| Mitzenmacher 2000. How Useful Is Old Information? IEEE TPDS | https://www.eecs.harvard.edu/~michaelm/abstracts/tpds2000.html | `mitzenmacher2000-old-information.pdf` | key | LR-14 |

## LLM request routing (`pdfs/llm-routing/`, notes `llm-routing.md`)

| Citation | URL | Local file | Depth | Used in |
|---|---|---|---|---|
| Tumkur et al. 2026. Calibrate, Then Route: A Measured Study of Learned Request Routing for Disaggregated LLM Serving. arXiv | https://arxiv.org/abs/2609.16206 | `tumkur2026-calibrate-then-route.pdf` (copy also in `eval-methodology/`) | full | LR-02, LR-04, LR-08, LR-12, LR-13, LR-14 |
| Lim et al. 2026. Lodestar: An Online-Learning LLM Inference Router. arXiv | https://arxiv.org/abs/2606.00946 | `lim2026-lodestar.pdf` | full | LR-06, LR-07; OOD clamp delta |
| Jain et al. 2024. Intelligent Router for LLM Workloads. arXiv (MSR) | https://arxiv.org/abs/2408.13510 | `jain2024-intelligent-router.pdf` | key | plan deltas (queue-threshold arm, prefill×decode feature) |
| Zhang et al. 2026. Simple is Better: Multiplication May Be All You Need for LLM Request Scheduling (LMetric). OSDI | https://arxiv.org/abs/2603.15202 | `zhang2026-lmetric.pdf` | key | LR-04, LR-07 |
| Wang et al. 2026. SMetric: Rethink LLM Scheduling for Serving Agents with Balanced Session-centric Scheduling. arXiv | https://arxiv.org/abs/2607.08565 | `wang2026-smetric.pdf` | key | LR-01, LR-04, LR-07, LR-08 |
| Ricci Toniolo et al. 2026. GORGO: Online Tuning for Cross-Region Network-Aware LLM Serving. arXiv | https://arxiv.org/abs/2602.11688 | `toniolo2026-gorgo.pdf` | full | LR-13 |
| Wu, Silwal & Zhang 2026. Randomization Boosts KV Caching, Learning Balances Query Load: A Joint Perspective. ICLR | https://arxiv.org/abs/2601.18999 | `wu2026-randomized-eviction-learned-routing.pdf` | key | risk 12 |
| Cheng 2026. CacheRoute: Planned Prefix-Affinity Routing for Large-Scale LLM Serving. arXiv | https://arxiv.org/abs/2608.19677 | `cheng2026-cacheroute.pdf` | full | LR-12 |
| Yuan et al. 2026. DualMap: Enabling Both Cache Affinity and Load Balancing for Distributed LLM Serving. ICLR | https://arxiv.org/abs/2602.06502 | `yuan2026-dualmap.pdf` | key | LR-04, LR-07 |
| Srivatsa et al. 2025. Preble: Efficient Distributed Prompt Scheduling for LLM Serving. ICLR | https://arxiv.org/abs/2407.00023 | `srivatsa2024-preble.pdf` | key | plan deltas (prefill×decode feature) |
| Kang et al. 2026. ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System. ICML | https://arxiv.org/abs/2602.13692 | `kang2026-thunderagent.pdf` | key | LR-14; plan deltas |
| Rajib, Zheng & Lou 2026. AgentServeSim: Serving-System Simulation and Policy Search for LLM Agent Programs. arXiv | https://arxiv.org/abs/2606.09613 | `rajib2026-agentservesim.pdf` | key | LR-02, LR-09 |
| Luo et al. 2025. Autellix: An Efficient Serving Engine for LLM Agents as General Programs | https://arxiv.org/abs/2502.13965 | none | web (§4.3) | background |
| Nixon et al. 2026. A Year in LLM Serving: Workload Evolution, Caching and Load-Balancing | https://arxiv.org/abs/2608.13573 | none | web (§7.2) | LR-13 metrics |
| llm-d project 2025. KV-cache wins you can see (precise prefix-cache-aware scheduling) | https://llm-d.ai/blog/kvcache-wins-you-can-see | none | web | LR-12, LR-14 |
| Jha et al. 2024. Learned Best-Effort LLM Serving | https://arxiv.org/abs/2401.07886 | none | abstract | tangential |
| Da & Kalyvianaki 2026. RouteBalance: Fused Model Routing and Load Balancing for Heterogeneous LLM Serving | https://arxiv.org/abs/2606.17949 | none | abstract | tangential |
| Cao et al. 2025. Locality-aware Fair Scheduling in LLM Serving (DLPM/D2LPM) | https://arxiv.org/abs/2501.14312 | none | abstract | out of scope |
| Li et al. 2025. Continuum: Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live. ICLR 2026 | https://arxiv.org/abs/2511.02230 | none | abstract | background |
| Ramjet (Helix load balancer). Not a paper; it is the basis of the campaign's `ramjet` port | https://github.com/helixml/ramjet | none | n/a | baseline provenance |

## Learned systems policies, black-box search, sim-to-real (`pdfs/learned-systems-policies/`, notes `learned-systems-policies.md`)

| Citation | URL | Local file | Depth | Used in |
|---|---|---|---|---|
| Mao, Schwarzkopf, Venkatakrishnan, Meng & Alizadeh 2019. Learning Scheduling Algorithms for Data Processing Clusters (Decima). SIGCOMM | https://arxiv.org/abs/1810.01963 | `mao2019-decima.pdf` | full | LR-01, LR-03, LR-10, LR-15 |
| Mao, Venkatakrishnan, Schwarzkopf & Alizadeh 2019. Variance Reduction for RL in Input-Driven Environments. ICLR | https://arxiv.org/abs/1807.02264 | `mao2019-input-driven-variance.pdf` | full | LR-03 |
| Mao et al. 2019. Park: An Open Platform for Learning-Augmented Computer Systems. NeurIPS | https://proceedings.neurips.cc/paper/2019/hash/f69e505b08403ad2298b9f262659929a-Abstract.html | `mao2019-park.pdf` | key | LR-07, LR-13 |
| Mao, Netravali & Alizadeh 2017. Neural Adaptive Video Streaming with Pensieve. SIGCOMM | https://web.mit.edu/pensieve/ | `mao2017-pensieve.pdf` | key | background (sim assumptions) |
| Yan et al. 2020. Learning in situ: a randomized experiment in video streaming (Puffer). NSDI | https://arxiv.org/abs/1906.01113 | `yan2020-puffer.pdf` (same paper as `eval-methodology/yan2020-puffer-learning-in-situ.pdf`) | key | LR-02, LR-14 |
| Xia, Zhou, Yan & Jiang 2022. Genet: Automatic Curriculum Generation for Learning Adaptation in Networking. SIGCOMM | https://arxiv.org/abs/2202.05940 | `xia2022-genet.pdf` | key | LR-11 |
| Alomar et al. 2023. CausalSim: A Causal Framework for Unbiased Trace-Driven Simulation. NSDI | https://arxiv.org/abs/2201.01811 | `alomar2023-causalsim.pdf` | key | LR-09 |
| Peng, Andrychowicz, Zaremba & Abbeel 2018. Sim-to-Real Transfer of Robotic Control with Dynamics Randomization. ICRA | https://arxiv.org/abs/1710.06537 | `peng2018-dynamics-randomization.pdf` | full | LR-14 |
| Mania, Guy & Recht 2018. Simple random search provides a competitive approach to RL (ARS). NeurIPS | https://arxiv.org/abs/1803.07055 | `mania2018-ars.pdf` | full | LR-05, LR-10 |
| Salimans, Ho, Chen, Sidor & Sutskever 2017. Evolution Strategies as a Scalable Alternative to RL. arXiv | https://arxiv.org/abs/1703.03864 | `salimans2017-es.pdf` | key | LR-03, LR-05 |
| Hansen 2016 (v2 2023). The CMA Evolution Strategy: A Tutorial. arXiv | https://arxiv.org/abs/1604.00772 | `hansen2016-cmaes-tutorial.pdf` | key | LR-05, LR-10 |
| Hansen, Niederberger, Guzzella & Koumoutsakos 2009. A Method for Handling Uncertainty in Evolutionary Optimization (UH-CMA-ES). IEEE TEC 13(1) | http://www.cmap.polytechnique.fr/~nikolaus.hansen/TEC2009.pdf | `hansen2009-uh-cmaes.pdf` | key | LR-01, LR-02, LR-03 |
| Ng & Jordan 2000. PEGASUS: A policy search method for large MDPs and POMDPs. UAI | https://arxiv.org/abs/1301.3878 | `ng2000-pegasus.pdf` | full | LR-03, LR-10 |
| Tahir, Cui & Koeppl 2022. Learning Mean-Field Control for Delayed Information Load Balancing in Large Queuing Systems. ICPP | https://arxiv.org/abs/2208.04777 | `tahir2022-meanfield-lb.pdf` | key | LR-12, LR-14 |
| van der Boor, Borst, van Leeuwaarden & Mukherjee 2022. Scalable load balancing in networked systems: a survey. Statistical Science | https://arxiv.org/abs/1806.05444 | `vanderboor2018-scalable-lb.pdf` | key | LR-12 |
| Zaheer et al. 2017. Deep Sets. NeurIPS | https://arxiv.org/abs/1703.06114 | `zaheer2017-deep-sets.pdf` | key | LR-06 |
| Lehman, Clune, Misevic et al. 2020. The Surprising Creativity of Digital Evolution. Artificial Life 26(2) | https://arxiv.org/abs/1803.03453 | `lehman2018-digital-evolution-creativity.pdf` | key | LR-13 |
| Heidrich-Meisner & Igel 2009. Racing for policy selection in CMA-ES. ICML | none | `heidrichmeisner2009-races-cmaes.FAILED-DOWNLOAD.html`: failed download, a bot-check page; list it in the cleanup ledger | not read | none |

## Evaluation methodology (`pdfs/eval-methodology/`, notes `eval-methodology.md`)

| Citation | URL | Local file | Depth | Used in |
|---|---|---|---|---|
| Zhong et al. 2024. DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving. OSDI | https://arxiv.org/abs/2401.09670 | `zhong2024-distserve.pdf` | key | LR-01, LR-08 |
| Schroeder, Wierman & Harchol-Balter 2006. Open Versus Closed: A Cautionary Tale. NSDI | https://www.cs.utoronto.ca/~bianca/papers/nsdi_camera.pdf | `schroeder2006-open-vs-closed.pdf` | key | LR-09, LR-12 |
| Wang et al. 2024 (v2 2025). Revisiting SLOs and System Level Metrics in LLM Serving. arXiv | https://arxiv.org/abs/2410.14257 | `wang2024-revisiting-slo-goodput.pdf` | key | LR-08, LR-13 |
| Agrawal et al. 2024. Vidur: A Large-Scale Simulation Framework for LLM Inference. MLSys | https://arxiv.org/abs/2405.05465 | `agrawal2024-vidur.pdf` | key | LR-14 |
| Henderson et al. 2018. Deep Reinforcement Learning that Matters. AAAI | https://arxiv.org/abs/1709.06560 | `henderson2018-deep-rl-matters.pdf` | key | LR-04, LR-10 |
| Agarwal et al. 2021. Deep RL at the Edge of the Statistical Precipice. NeurIPS | https://arxiv.org/abs/2108.13264 | `agarwal2021-statistical-precipice.pdf` | key | LR-01, LR-11 |
| Demšar 2006. Statistical Comparisons of Classifiers over Multiple Data Sets. JMLR 7 | https://www.jmlr.org/papers/v7/demsar06a.html | `demsar2006-statistical-comparisons.pdf` | key | LR-01, LR-11 |
| Cawley & Talbot 2010. On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. JMLR 11 | https://www.jmlr.org/papers/v11/cawley10a.html | `cawley2010-overfitting-model-selection.pdf` | key | LR-10; N=6 split delta |
| Dodge et al. 2019. Show Your Work: Improved Reporting of Experimental Results. EMNLP | https://arxiv.org/abs/1909.03004 | `dodge2019-show-your-work.pdf` | key | LR-10 |
| Bouthillier et al. 2021. Accounting for Variance in Machine Learning Benchmarks. MLSys | https://arxiv.org/abs/2103.03098 | `bouthillier2021-variance-ml-benchmarks.pdf` | key | LR-02, LR-11 |
| Melis, Dyer & Blunsom 2018. On the State of the Art of Evaluation in Neural Language Models. ICLR | https://arxiv.org/abs/1707.05589 | none | abstract | LR-04 |
| Agrawal et al. 2024. Etalon: Holistic Performance Evaluation Framework for LLM Inference Systems | https://arxiv.org/abs/2407.07000 | none | abstract | background |
| Krishnamachari 2026. How to Do Statistical Evaluations in ECE/CS Papers: A Practical Playbook | https://arxiv.org/abs/2605.00428 | none | web (§8, §11, §14, §17, §20) | LR-02, LR-11 |
| Abdelfattah et al. 2026. Load Testing for Machine Learning Model Serving Systems at Scale | https://arxiv.org/abs/2606.22013 | none | abstract | LR-01 (warm-up) |
| Berger 1982. Multiparameter hypothesis testing and acceptance sampling. Technometrics | none | none | from memory; not read | LR-11 (intersection-union test) |

## Duplicates and oddities

These are listed for the campaign cleanup ledger. Nothing was deleted.

- `pdfs/eval-methodology/tumkur2026-calibrate-then-route.pdf` is the same file as
  `pdfs/llm-routing/tumkur2026-calibrate-then-route.pdf` (4,387,625 bytes each).
- `pdfs/eval-methodology/yan2020-puffer-learning-in-situ.pdf` and
  `pdfs/learned-systems-policies/yan2020-puffer.pdf` are the same paper.
- `pdfs/dcm-context/train2009-gev-ch4.pdf` is MacBinary-wrapped. Use `train2009-gev-ch4.clean.pdf`.
- `pdfs/dcm-context/jain2022-agile-action-relations.pdf` is a slide deck (7.7 MB), not the paper.
- `pdfs/learned-systems-policies/heidrichmeisner2009-races-cmaes.FAILED-DOWNLOAD.html` is an HTML
  bot-check page.
