/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Customer and partner publications about Dynamo, hosted on third-party sites.
 *
 * Link-only by design: these are other organisations' articles, so the entry
 * carries the title, the publisher and the date, and sends the reader to the
 * source. Nothing is mirrored or embedded -- most of these publishers send
 * X-Frame-Options or a restrictive frame-ancestors, so an embed would render
 * an empty box for roughly two thirds of the list even if we wanted one.
 *
 * `date` is the publisher's date as printed on the article; `iso` is a sort key
 * only, so a month-only date is normalised to the first of the month and the
 * one undated entry sorts last. Keep the array in reverse-chronological order.
 *
 * Logos key off `partner`, not the host of `url`: several of these are
 * published somewhere other than the author's own site (Doubleword on
 * TelecomTV, Deloitte on LinkedIn), and keying on the host credited the wrong
 * company. A partner absent from PUBLISHER_LOGOS falls back to its initials.
 */

export interface Publication {
  title: string;
  url: string;
  /** Publishing organisation, as it should be shown to readers. */
  partner: string;
  /** Human-readable date exactly as published, or "Undated". */
  date: string;
  /** Sort key, YYYY-MM-DD; the component orders on this. Null when undated. */
  iso: string | null;
  /** The publisher has since revised the article; shown next to the date. */
  updated?: boolean;
}

export const PUBLICATIONS: Publication[] = [
  {
    title: "How We Cut Inference Cold Start to Seconds With NVIDIA Dynamo",
    url: "https://www.photoroom.com/inside-photoroom/how-we-cut-gpu-cold-starts-from-minutes-to-seconds-with-memory-checkpointing",
    partner: "Photoroom",
    date: "Jul 23, 2026",
    iso: "2026-07-23",
  },
  {
    title: "NVIDIA Vera Rubin NVL72 on CoreWeave: 10x More Tokens per Megawatt",
    url: "https://coreweave.com/blog/nvidia-vera-rubin-nvl72-on-coreweave-10x-more-tokens-per-megawatt-than-blackwell",
    partner: "CoreWeave",
    date: "Jul 21, 2026",
    iso: "2026-07-21",
  },
  {
    title: "Gcore Introduces Global Inference Routing Accelerated by NVIDIA Dynamo",
    url: "https://gcore.com/blog/global-inference-routing",
    partner: "Gcore",
    date: "Jul 20, 2026",
    iso: "2026-07-20",
  },
  {
    title: "SWE-1.7: Frontier Intelligence at a Fraction of the Cost",
    url: "https://cognition.com/blog/swe-1-7",
    partner: "Cognition",
    date: "Jul 8, 2026",
    iso: "2026-07-08",
  },
  {
    title: "Booting Fast and Slow",
    url: "https://hcompany.ai/booting-fast-and-slow",
    partner: "H Company",
    date: "Jul 8, 2026",
    iso: "2026-07-08",
  },
  {
    title: "Introducing and a Deep Dive Into Dynamo with vCluster",
    url: "https://www.vcluster.com/blog/nvidia-dynamo-with-vcluster",
    partner: "vCluster",
    date: "Jul 7, 2026",
    iso: "2026-07-07",
  },
  {
    title: "Lowering Multimodal Inference Cost with Heterogeneous E/PD Disaggregation",
    url: "https://community.intel.com/t5/Blogs/Tech-Innovation/Data-Center/Lowering-Multimodal-Inference-Cost-with-Heterogeneous-E-PD/post/1752499",
    partner: "Intel",
    date: "Jun 26, 2026",
    iso: "2026-06-26",
  },
  {
    title: "How We Built the World's Fastest API for GLM-5.2",
    url: "https://www.baseten.co/blog/how-we-built-the-worlds-fastest-api-for-glm-52/",
    partner: "Baseten",
    date: "Jun 23, 2026",
    iso: "2026-06-23",
  },
  {
    title: "RL at 1T Scale: prime-rl Performance Deep Dive",
    url: "https://www.primeintellect.ai/blog/rl-at-1t-scale",
    partner: "Prime Intellect",
    date: "Jun 21, 2026",
    iso: "2026-06-21",
  },
  {
    title: "Keynote: Plug in and Scale: Serving LLM Models on Kubernetes Made Simple",
    url: "https://kccncind2026.sched.com/event/2IW3S",
    partner: "AstraZeneca",
    date: "Jun 18, 2026",
    iso: "2026-06-18",
  },
  {
    title: "Deploy a Dynamo Inference Service with PD Disaggregation",
    url: "https://www.alibabacloud.com/help/en/ack/cloud-native-ai-suite/user-guide/deploy-dynamo-pd-separated-inference-services",
    partner: "Alibaba Cloud / ACK",
    date: "Jun 15, 2026",
    iso: "2026-06-15",
  },
  {
    title: "Deploying NVIDIA Dynamo PD Disaggregation With dstack",
    url: "https://dstack.ai/blog/nvidia-dynamo/",
    partner: "dstack",
    date: "Jun 10, 2026",
    iso: "2026-06-10",
  },
  {
    title: "How the UK Is Turning Sovereign AI Ambition Into Action With NVIDIA Technologies",
    url: "https://www.telecomtv.com/content/digital-platforms-services/how-the-uk-is-turning-sovereign-ai-ambition-into-action-with-nvidia-technologies-55622/",
    partner: "Doubleword",
    date: "Jun 8, 2026",
    iso: "2026-06-08",
  },
  {
    title: "What Is Context Memory? The New Infrastructure Layer Powering AI Inference",
    url: "https://www.weka.io/learn/ai-ml/what-is-context-memory-primary/",
    partner: "WEKA",
    date: "May 28, 2026",
    iso: "2026-05-28",
  },
  {
    title: "NVIDIA Dynamo on AKS - Autoscaling LLM Inference",
    url: "https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/nvidia-dynamo-on-aks---autoscaling-llm-inference/4499545",
    partner: "Microsoft Azure",
    date: "May 13, 2026",
    iso: "2026-05-13",
  },
  {
    title: "Serving DeepSeek-V4: Why Million-Token Context Is an Inference Systems Problem",
    url: "https://www.together.ai/blog/serving-deepseek-v4-why-million-token-context-is-an-inference-systems-problem",
    partner: "Together AI",
    date: "May 11, 2026",
    iso: "2026-05-11",
  },
  {
    title: "NVIDIA Dynamo on AKS: Disaggregated LLM Inference with H100 GPUs",
    url: "https://azureglobalblackbelts.com/2026/05/08/nvidia-dynamo-on-aks/",
    partner: "Azure Global Black Belt",
    date: "May 8, 2026",
    iso: "2026-05-08",
  },
  {
    title: "GB200 NVL72 vs B200 on Kimi K2.5: 3.1x from Wide EP vLLM",
    url: "https://inferencex.semianalysis.com/blog/gb200-nvl72-kimi-k2-5-vllm-wide-ep-3x-vs-b200",
    partner: "SemiAnalysis / InferenceX",
    date: "Apr 23, 2026",
    iso: "2026-04-23",
  },
  {
    title: "Amazon SageMaker AI Now Supports Optimized Generative AI Inference Recommendations",
    url: "https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-ai-now-supports-optimized-generative-ai-inference-recommendations/",
    partner: "AWS",
    date: "Apr 22, 2026",
    iso: "2026-04-22",
  },
  {
    title: "ClearML + NVIDIA Dynamo: A Production Control Plane for Distributed AI Inference at Scale",
    url: "https://clear.ml/blog/clearml-nvidia-dynamo-a-production-control-plane-for-distributed-ai-inference-at-scale",
    partner: "ClearML",
    date: "Apr 9, 2026",
    iso: "2026-04-09",
  },
  {
    title: "NVIDIA Dynamo: Turning Disaggregated Inference Into a Production System",
    url: "https://rafay.co/ai-and-cloud-native-blog/nvidia-dynamo-turning-disaggregated-inference-into-a-production-system",
    partner: "Rafay",
    date: "Apr 7, 2026",
    iso: "2026-04-07",
  },
  {
    title: "Introduction to Disaggregated Inference: Why It Matters",
    url: "https://rafay.co/ai-and-cloud-native-blog/introduction-to-disaggregated-inference-why-it-matters",
    partner: "Rafay",
    date: "Mar 29, 2026",
    iso: "2026-03-29",
  },
  {
    title: "NVIDIA Dynamo 1.0: Disaggregated LLM Inference Deployment Guide (2026)",
    url: "https://www.spheron.network/blog/nvidia-dynamo-disaggregated-inference-guide/",
    partner: "Spheron",
    date: "Mar 25, 2026",
    iso: "2026-03-25",
  },
  {
    title: "San Jose GTC 2026 Wrap-Up",
    url: "https://www.gmicloud.ai/en/blog/san-jose-gtc-2026-wrap-up",
    partner: "GMI Cloud",
    date: "Mar 24, 2026",
    iso: "2026-03-24",
  },
  {
    title: "NVIDIA Dynamo 1.0 Is Now Available to DigitalOcean Customers",
    url: "https://www.digitalocean.com/blog/nvidia-dynamo-1-now-available",
    partner: "DigitalOcean",
    date: "Mar 19, 2026",
    iso: "2026-03-19",
  },
  {
    title: "Infrastructure for Enterprise AI Inference With Vultr, DDN, NVIDIA Dynamo and Nemotron",
    url: "https://blogs.vultr.com/NVIDIA-Dynamo-Nemotron-DDN",
    partner: "Vultr",
    date: "Mar 17, 2026",
    iso: "2026-03-17",
  },
  {
    title: "How Rafay and NVIDIA Help Neoclouds Monetize Accelerated Computing",
    url: "https://docs.rafay.co/blog/2026/03/17/how-rafay-and-nvidia-help-neoclouds-monetize-accelerated-computing-with-token-factories/",
    partner: "Rafay",
    date: "Mar 17, 2026",
    iso: "2026-03-17",
  },
  {
    title: "Running GenAI at Amazon Ads Scale: Lessons from Building and Operating LLM Inference Systems",
    url: "https://www.nvidia.com/ja-jp/on-demand/session/gtc26-s81656/",
    partner: "Amazon Ads",
    date: "Mar 16, 2026",
    iso: "2026-03-16",
  },
  {
    title: "LMCache + NVIDIA Dynamo 1.0: A Match Made in Inference Heaven",
    url: "https://blog.lmcache.ai/en/2026/03/16/lmcache-nvidia-dynamo-1-0-a-match-made-in-inference-heaven/",
    partner: "LMCache",
    date: "Mar 16, 2026",
    iso: "2026-03-16",
  },
  {
    title: "2x Faster Inference With KV Cache-Aware Routing",
    url: "https://www.baseten.co/blog/how-baseten-achieved-2x-faster-inference-with-nvidia-dynamo/",
    partner: "Baseten",
    date: "Mar 16, 2026",
    iso: "2026-03-16",
  },
  {
    title: "Reducing TTFT by CPUMaxxing Tokenization",
    url: "https://www.crusoe.ai/resources/blog/reducing-ttft-by-cpumaxxing-tokenization",
    partner: "Crusoe",
    date: "Mar 16, 2026",
    iso: "2026-03-16",
  },
  {
    title: "Crusoe Expands NVIDIA Collaboration Across the Full AI Factory Stack",
    url: "https://www.crusoe.ai/resources/newsroom/crusoe-expands-nvidia-collaboration",
    partner: "Crusoe",
    date: "Mar 16, 2026",
    iso: "2026-03-16",
  },
  {
    title: "GMI Cloud Supports NVIDIA Dynamo 1.0 and OpenShell",
    url: "https://www.gmicloud.ai/ja/blog/gmi-cloud-joins-nvidias-dynamo-1-0-launch-at-gtc-2026",
    partner: "GMI Cloud",
    date: "Mar 16, 2026",
    iso: "2026-03-16",
  },
  {
    title: "Google Cloud AI Infrastructure at NVIDIA GTC 2026",
    url: "https://cloud.google.com/blog/products/compute/google-cloud-ai-infrastructure-at-nvidia-gtc-2026",
    partner: "Google Cloud",
    date: "Mar 16, 2026",
    iso: "2026-03-16",
  },
  {
    title: "Together AI at NVIDIA GTC 2026",
    url: "https://www.together.ai/blog/together-ai-at-nvidia-gtc-2026",
    partner: "Together AI",
    date: "Mar 16, 2026",
    iso: "2026-03-16",
  },
  {
    title: "How DigitalOcean's Agentic Inference Cloud Achieved 67% Lower Inference Costs for Workato",
    url: "https://www.digitalocean.com/blog/workato-nvidia-technical-deep-dive-agentic-inference-cloud",
    partner: "DigitalOcean / Workato",
    date: "Mar 4, 2026",
    iso: "2026-03-04",
    updated: true,
  },
  {
    title: "Inference Platform Overview",
    url: "https://developer.aliyun.com/article/1713909",
    partner: "Alibaba Cloud community",
    date: "Mar 1, 2026",
    iso: "2026-03-01",
  },
  {
    title: "Deloitte Drives AI Transformations With NVIDIA Dynamo and Nemotron",
    url: "https://www.linkedin.com/posts/sanghamitra-pati-0528b211_deloitte-expands-ai-factory-capabilities-activity-7439759386241150976-i7mM",
    partner: "Deloitte",
    date: "Mar 2026",
    iso: "2026-03-01",
  },
  {
    title: "Deployment of NVIDIA Dynamo",
    url: "https://docs.opennebula.io/7.0/solutions/ai_factory_blueprints/containerized_ai_execution/nvidia_dynamo/",
    partner: "OpenNebula",
    date: "Mar 2026",
    iso: "2026-03-01",
  },
  {
    title: "Introducing Faster, Lower-Cost LLM Inference With NVIDIA Dynamo",
    url: "https://gcore.com/blog/dynamo",
    partner: "Gcore",
    date: "Feb 25, 2026",
    iso: "2026-02-25",
  },
  {
    title: "Gcore Integrates NVIDIA Dynamo as a Fully Managed Service",
    url: "https://gcore.com/press-releases/nvidia-dynamo-on-gcore",
    partner: "Gcore",
    date: "Feb 24, 2026",
    iso: "2026-02-24",
  },
  {
    title: "Unlocking 25x Inference Performance with SGLang on NVIDIA GB300 NVL72",
    url: "https://www.lmsys.org/blog/2026-02-20-gb300-inferencex/",
    partner: "LMSYS / SGLang",
    date: "Feb 20, 2026",
    iso: "2026-02-20",
  },
  {
    title: "The Baseten Inference Stack at NVIDIA Dynamo Day",
    url: "https://www.baseten.co/blog/nvidia-dynamo-day-baseten-inference-stack/",
    partner: "Baseten",
    date: "Feb 3, 2026",
    iso: "2026-02-03",
  },
  {
    title: "CoreWeave Achieves NVIDIA Exemplar Cloud Validation for Inference",
    url: "https://www.coreweave.com/blog/coreweave-becomes-one-of-the-first-cloud-providers-to-achieve-nvidia-exemplar-cloud-validation-for-inference-on-nvidia-gb200-nvl72",
    partner: "CoreWeave",
    date: "Jan 28, 2026",
    iso: "2026-01-28",
  },
  {
    title: "Scaling WideEP Mixture-of-Experts Inference with Google Cloud A4X (GB200) and NVIDIA Dynamo",
    url: "https://cloud.google.com/blog/products/compute/scaling-moe-inference-with-nvidia-dynamo-on-google-cloud-a4x",
    partner: "Google Cloud",
    date: "Jan 22, 2026",
    iso: "2026-01-22",
  },
  {
    title: "Supercharging AI Inference: Pure KVA Now Integrates with NVIDIA Dynamo for Scalable, Low-Latency LLM Inference",
    url: "https://blog.everpuredata.com/news-events/pure-kva-integrates-nvidia-dynamo-for-scalable-low-latency-llm-inference/",
    partner: "Everpure / Pure Storage",
    date: "Jan 14, 2026",
    iso: "2026-01-14",
  },
  {
    title: "NVIDIA Dynamo + VAST = Scalable, Optimized Inference",
    url: "https://www.vastdata.com/blog/nvidia-dynamo-vast-scalable-optimized-inference",
    partner: "VAST Data",
    date: "Dec 16, 2025",
    iso: "2025-12-16",
  },
  {
    title: "Disaggregated Inference: 18 Months Later",
    url: "https://haoailab.com/blogs/distserve-retro/",
    partner: "Hao AI Lab / UCSD",
    date: "Nov 3, 2025",
    iso: "2025-11-03",
  },
  {
    title: "Scaling Multi-Node LLM Inference with NVIDIA Dynamo and ND GB200 NVL72 GPUs on AKS",
    url: "https://blog.aks.azure.com/2025/10/24/dynamo-on-aks",
    partner: "Microsoft Azure / AKS",
    date: "Oct 24, 2025",
    iso: "2025-10-24",
  },
  {
    title: "Fast and Efficient AI Inference with New NVIDIA Dynamo Recipe on AI Hypercomputer",
    url: "https://cloud.google.com/blog/products/compute/ai-inference-recipe-using-nvidia-dynamo-with-ai-hypercomputer",
    partner: "Google Cloud",
    date: "Sep 10, 2025",
    iso: "2025-09-10",
  },
  {
    title: "Accelerate Generative AI Inference with NVIDIA Dynamo and Amazon EKS",
    url: "https://aws.amazon.com/blogs/machine-learning/accelerate-generative-ai-inference-with-nvidia-dynamo-and-amazon-eks/",
    partner: "AWS",
    date: "Jul 15, 2025",
    iso: "2025-07-15",
  },
  {
    title: "Run Nvidia Dynamo on Any Cloud or Kubernetes with SkyPilot",
    url: "https://docs.skypilot.ai/en/latest/examples/serving/nvidia-dynamo.html",
    partner: "SkyPilot",
    date: "Undated",
    iso: null,
  },
];
