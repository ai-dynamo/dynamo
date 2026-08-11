/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Papers that use, extend, or benchmark Dynamo, hosted on arXiv, ACM, USENIX,
 * MLSys and elsewhere. Link-only, like the partner publications beside them.
 *
 * Titles and dates are taken from the source itself -- the arXiv abstract page,
 * Crossref for the DOI, or the first page of the PDF -- rather than retyped, so
 * they match what a reader lands on.
 *
 * `org` is the affiliation to show, and is empty where the paper is listed by
 * venue alone; `venue` always carries something, so the card falls back to it.
 * Keep the array in reverse-chronological order.
 */

export interface ResearchPaper {
  title: string;
  url: string;
  /** Affiliation to credit, or "" to fall back to the venue. */
  org: string;
  /** Publication venue: arXiv, MLSys 2026, OSDI '26, and so on. */
  venue: string;
  /** Human-readable date, or a bare year where the venue gives no day. */
  date: string;
  /** Sort key, YYYY-MM-DD. Year-only entries are pinned to January. */
  iso: string;
}

export const RESEARCH_PAPERS: ResearchPaper[] = [
  {
    title: "ARK: Avoiding Routing Collisions for KV Cache Transfer in Disaggregated LLM Inference",
    url: "https://saeed.github.io/files/arc_niac26.pdf",
    org: "Georgia Tech",
    venue: "SIGCOMM '26 NIAC",
    date: "Aug 2026",
    iso: "2026-08-17",
  },
  {
    title: "A Photonic-CXL Memory Appliance for Scalable KV Cache Management in LLM Inference",
    url: "https://arxiv.org/abs/2607.27187",
    org: "Marvell",
    venue: "arXiv",
    date: "Jul 29, 2026",
    iso: "2026-07-29",
  },
  {
    title: "Revisiting Pipeline Parallelism for LLM Serving",
    url: "https://www.usenix.org/system/files/osdi26-hwang.pdf",
    org: "Korea University",
    venue: "OSDI '26",
    date: "Jul 13, 2026",
    iso: "2026-07-13",
  },
  {
    title: "Solyx AI Grid: Hardware-Telemetry-Aware Routing Across Geographically Distributed GPU Clusters",
    url: "https://arxiv.org/abs/2606.15050",
    org: "Solyx AI",
    venue: "arXiv",
    date: "Jun 13, 2026",
    iso: "2026-06-13",
  },
  {
    title: "The Price of Anarchy in Disaggregated Inference",
    url: "https://arxiv.org/abs/2606.17081",
    org: "",
    venue: "arXiv",
    date: "Jun 11, 2026",
    iso: "2026-06-11",
  },
  {
    title: "Breaking the Ice: Analyzing Cold Start Latency in vLLM",
    url: "https://proceedings.mlsys.org/paper_files/paper/2026/file/29416b66c2149872b9d1415a3fd2c5e0-Paper-Conference.pdf",
    org: "",
    venue: "MLSys 2026",
    date: "May 2026",
    iso: "2026-05-18",
  },
  {
    title: "A Pragmatic Exploration of Prefill-Decode Disaggregation in Large Scale Inference",
    url: "https://proceedings.mlsys.org/paper_files/paper/2026/file/d49cee5f3a79d97d719df255689d83d7-Paper-Conference.pdf",
    org: "NVIDIA",
    venue: "MLSys 2026",
    date: "May 2026",
    iso: "2026-05-18",
  },
  {
    title: "Adaptive Parallelism for LLM Inference with Model Irrelevant Profiler",
    url: "https://ieeexplore.ieee.org/document/11581475",
    org: "Lenovo",
    venue: "Frontiers of Computer Science",
    date: "May 9, 2026",
    iso: "2026-05-09",
  },
  {
    title: "A Case for a Simulation-Driven Exploration of Distributed GenAI Platforms",
    url: "https://dl.acm.org/doi/10.1145/3805621.3807623",
    org: "IBM Research",
    venue: "EuroMLSys '26",
    date: "Apr 27, 2026",
    iso: "2026-04-27",
  },
  {
    title: "Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter",
    url: "https://arxiv.org/abs/2604.15039",
    org: "",
    venue: "arXiv",
    date: "Apr 16, 2026",
    iso: "2026-04-16",
  },
  {
    title: "NCCL EP: Towards a Unified Expert Parallel Communication API for NCCL",
    url: "https://arxiv.org/abs/2603.13606",
    org: "",
    venue: "arXiv",
    date: "Mar 13, 2026",
    iso: "2026-03-13",
  },
  {
    title: "Efficient Multi-round LLM Inference over Disaggregated Serving",
    url: "https://arxiv.org/abs/2602.14516",
    org: "",
    venue: "ICML 2026",
    date: "Feb 16, 2026",
    iso: "2026-02-16",
  },
  {
    title: "ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System",
    url: "https://arxiv.org/abs/2602.13692",
    org: "NVIDIA",
    venue: "arXiv",
    date: "Feb 14, 2026",
    iso: "2026-02-14",
  },
  {
    title: "AIConfigurator: Lightning-Fast Configuration Optimization for Multi-Framework LLM Serving",
    url: "https://arxiv.org/abs/2601.06288",
    org: "NVIDIA",
    venue: "arXiv",
    date: "Jan 9, 2026",
    iso: "2026-01-09",
  },
  {
    title: "Optimizing GPU Workloads on Kubernetes: An Integrated Approach Using NVIDIA Dynamo, run:ai (KAI), and Amazon EKS",
    url: "https://ieeexplore.ieee.org/document/11609238",
    org: "Amazon",
    venue: "IEEE Access",
    date: "2026",
    iso: "2026-01-01",
  },
  {
    title: "TraCT: Disaggregated LLM Serving with CXL Shared Memory KV Cache at Rack-Scale",
    url: "https://arxiv.org/abs/2512.18194",
    org: "SK Hynix",
    venue: "arXiv",
    date: "Dec 20, 2025",
    iso: "2025-12-20",
  },
  {
    title: "DuetServe: Harmonizing Prefill and Decode for LLM Serving via Adaptive GPU Multiplexing",
    url: "https://arxiv.org/abs/2511.04791",
    org: "",
    venue: "arXiv",
    date: "Nov 6, 2025",
    iso: "2025-11-06",
  },
  {
    title: "AMD MI300X GPU Performance Analysis",
    url: "https://arxiv.org/abs/2510.27583",
    org: "Celestica AI",
    venue: "arXiv",
    date: "Oct 31, 2025",
    iso: "2025-10-31",
  },
  {
    title: "From Attention to Disaggregation: Tracing the Evolution of LLM Inference",
    url: "https://arxiv.org/abs/2511.07422",
    org: "Capital One",
    venue: "arXiv",
    date: "Oct 16, 2025",
    iso: "2025-10-16",
  },
  {
    title: "BanaServe: Unified KV Cache and Dynamic Module Migration for Balancing Disaggregated LLM Serving in AI Infrastructure",
    url: "https://arxiv.org/abs/2510.13223",
    org: "",
    venue: "arXiv",
    date: "Oct 15, 2025",
    iso: "2025-10-15",
  },
  {
    title: "LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference",
    url: "https://arxiv.org/abs/2510.09665",
    org: "LMCache",
    venue: "arXiv",
    date: "Oct 8, 2025",
    iso: "2025-10-08",
  },
  {
    title: "GreenLLM: SLO-Aware Dynamic Frequency Scaling for Energy-Efficient LLM Serving",
    url: "https://arxiv.org/abs/2508.16449",
    org: "EPFL",
    venue: "arXiv",
    date: "Aug 22, 2025",
    iso: "2025-08-22",
  },
  {
    title: "Toward Disaggregated and Heterogenous AI Systems",
    url: "https://ieeexplore.ieee.org/document/11072015",
    org: "",
    venue: "IEEE Micro",
    date: "May 2025",
    iso: "2025-05-01",
  },
  {
    title: "semi-PD: Towards Efficient LLM Serving via Phase-Wise Disaggregated Computation and Unified Storage",
    url: "https://arxiv.org/abs/2504.19867",
    org: "Infinigence-AI",
    venue: "arXiv",
    date: "Apr 28, 2025",
    iso: "2025-04-28",
  },
];
