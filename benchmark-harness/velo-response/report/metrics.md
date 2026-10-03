# Tyche Velo response comparison

Values are medians of four balanced runs. The CSV includes each range and changes against both TCP and QUIC.

| Metric | TCP | QUIC | Velo TCP | Velo RDMA |
|---|---:|---:|---:|---:|
| requests_per_second | 2475.602 | 4581.617 | 4241.240 | 4259.370 |
| output_tokens_per_second | 2338099.818 | 4239760.374 | 3893506.740 | 3900621.146 |
| frontend_cpu_ms_per_request | 40.821 | 27.035 | 32.010 | 32.031 |
| frontend_cpu_us_per_output_token | 43.161 | 29.241 | 35.020 | 34.940 |
| frontend_mean_cores | 100.989 | 124.555 | 136.257 | 136.875 |
| frontend_system_cores | 53.240 | 36.511 | 38.888 | 37.290 |
| frontend_ucx_progress_cores | 0.000 | 0.000 | 0.000 | 1.791 |
| host_system_cores | 22.902 | 20.852 | 22.490 | 21.909 |
| host_softirq_cores | 34.035 | 17.198 | 18.024 | 16.843 |
| ethernet_rx_host_packets_per_second | 3608499.898 | 2986871.230 | 3215040.029 | 2509201.367 |
| ethernet_tx_host_packets_per_second | 2105915.096 | 2472268.135 | 2513481.480 | 2471094.977 |
| time_to_first_token_p50_ms | 1546.345 | 90.433 | 74.032 | 79.126 |
| time_to_first_token_p95_ms | 2453.492 | 898.490 | 1298.012 | 1296.453 |
| time_to_first_token_p99_ms | 5380.350 | 975.664 | 1420.230 | 1384.406 |
| inter_token_latency_p50_ms | 0.112 | 0.648 | 0.624 | 0.621 |
| inter_token_latency_p95_ms | 1.114 | 2.001 | 2.010 | 1.985 |
| inter_token_latency_p99_ms | 1.748 | 2.804 | 2.762 | 2.750 |
| request_latency_p50_ms | 1635.938 | 824.574 | 1160.581 | 1176.328 |
| request_latency_p95_ms | 5373.005 | 4744.141 | 4830.071 | 4843.495 |
| request_latency_p99_ms | 11551.079 | 11678.878 | 11788.451 | 11674.923 |
| ethernet_rx_wire_packets_per_second | 3608402.871 | 2986774.779 | 3215015.853 | 2509091.422 |
| ethernet_rx_wire_packets_per_request | 1461.359 | 650.328 | 756.939 | 590.628 |
| ethernet_rx_wire_packets_per_output_token | 1.546 | 0.704 | 0.826 | 0.642 |
| rdma_rx_wire_packets_per_second | 0.941 | 0.776 | 0.946 | 888139.749 |
| rdma_rx_wire_packets_per_request | 0.000 | 0.000 | 0.000 | 208.367 |
| rdma_rx_wire_packets_per_output_token | 0.000 | 0.000 | 0.000 | 0.227 |
| total_rx_wire_packets_per_second | 3608403.645 | 2986775.387 | 3215016.623 | 3395620.764 |
| total_rx_wire_packets_per_request | 1461.359 | 650.328 | 756.939 | 798.995 |
| total_rx_wire_packets_per_output_token | 1.546 | 0.704 | 0.826 | 0.869 |
| ethernet_tx_wire_packets_per_second | 2105867.403 | 2472207.132 | 2513477.813 | 2471044.471 |
| ethernet_tx_wire_packets_per_request | 857.627 | 538.865 | 591.531 | 579.016 |
| ethernet_tx_wire_packets_per_output_token | 0.908 | 0.583 | 0.646 | 0.630 |
| rdma_tx_wire_packets_per_second | 0.936 | 0.779 | 0.937 | 846056.841 |
| rdma_tx_wire_packets_per_request | 0.000 | 0.000 | 0.000 | 198.540 |
| rdma_tx_wire_packets_per_output_token | 0.000 | 0.000 | 0.000 | 0.216 |
| total_tx_wire_packets_per_second | 2105868.173 | 2472207.744 | 2513478.583 | 3314575.128 |
| total_tx_wire_packets_per_request | 857.627 | 538.866 | 591.531 | 777.370 |
| total_tx_wire_packets_per_output_token | 0.908 | 0.583 | 0.646 | 0.845 |
| total_wire_packets_per_second | 5714271.817 | 5458983.131 | 5728495.206 | 6710195.892 |
| total_wire_packets_per_request | 2318.986 | 1189.431 | 1348.064 | 1576.365 |
| total_wire_packets_per_output_token | 2.454 | 1.287 | 1.471 | 1.714 |
| ethernet_mean_rx_wire_bytes | 437.724 | 897.509 | 790.863 | 727.750 |
| ethernet_rx_256_to_511_bytes_phy_percent | 12.489 | 24.636 | 25.145 | 26.677 |
| ethernet_rx_8192_to_10239_bytes_phy_percent | 0.000 | 0.000 | 0.000 | 0.000 |
| ethernet_rx_65_to_127_bytes_phy_percent | 14.030 | 20.883 | 25.006 | 31.250 |
| ethernet_rx_2048_to_4095_bytes_phy_percent | 0.000 | 0.000 | 0.000 | 0.000 |
| ethernet_rx_64_bytes_phy_percent | 0.000 | 0.000 | 0.000 | 0.000 |
| ethernet_rx_4096_to_8191_bytes_phy_percent | 0.000 | 0.000 | 0.000 | 0.000 |
| ethernet_rx_512_to_1023_bytes_phy_percent | 0.928 | 2.779 | 6.975 | 2.691 |
| ethernet_rx_128_to_255_bytes_phy_percent | 56.012 | 0.108 | 0.560 | 0.017 |
| ethernet_rx_1519_to_2047_bytes_phy_percent | 0.000 | 0.000 | 0.000 | 0.000 |
| ethernet_rx_1024_to_1518_bytes_phy_percent | 16.495 | 51.594 | 42.326 | 39.273 |
| rx_discards_phy_per_second | 0.000 | 0.000 | 0.000 | 0.000 |
| tx_discards_phy_per_second | 0.000 | 0.000 | 0.000 | 0.000 |
| rx_out_of_buffer_per_second | 0.204 | 0.042 | 0.000 | 0.000 |
| RetransSegs_per_second | 437.061 | 4406.624 | 855.685 | 689.017 |
| rdma_port_rcv_errors_per_second | 0.000 | 0.000 | 0.000 | 0.000 |
| rdma_port_xmit_discards_per_second | 0.000 | 0.000 | 0.000 | 0.000 |
| rdma_out_of_buffer_per_second | 0.000 | 0.000 | 0.000 | 0.000 |
| rdma_packet_seq_err_per_second | 0.000 | 0.000 | 0.000 | 0.000 |
| rdma_local_ack_timeout_err_per_second | 0.000 | 0.000 | 0.000 | 0.000 |
| rdma_req_transport_retries_exceeded_per_second | 0.000 | 0.000 | 0.000 | 0.000 |
| frontend_mean_rss_gib | 42.734 | 45.973 | 39.193 | 39.209 |
| client_mean_cores | 81.356 | 129.787 | 127.542 | 127.477 |

Provisional limits against TCP:

- tcp: pass
- quic: fail: inter_token_latency_p99_ms
- velo-tcp: fail: inter_token_latency_p99_ms
- velo-rdma: fail: inter_token_latency_p99_ms

Small error counts are retained in this directional comparison.

| Run | Exported completed records | Errors | Error fraction | Cancelled at drain deadline | Client event-loop warnings |
|---|---:|---:|---:|---:|---:|
| r1-tcp | 391480 | 0 | 0.000000% | 1589 | 3212 |
| r2-tcp | 395172 | 0 | 0.000000% | 1571 | 7448 |
| r3-tcp | 386631 | 0 | 0.000000% | 1647 | 4078 |
| r4-tcp | 391647 | 0 | 0.000000% | 1429 | 6664 |
| r1-quic | 720739 | 0 | 0.000000% | 0 | 34086 |
| r2-quic | 739413 | 0 | 0.000000% | 0 | 36056 |
| r3-quic | 744957 | 0 | 0.000000% | 0 | 40014 |
| r4-quic | 721241 | 1 | 0.000139% | 0 | 34504 |
| r1-velo-tcp | 667712 | 0 | 0.000000% | 0 | 28232 |
| r2-velo-tcp | 668389 | 0 | 0.000000% | 0 | 31494 |
| r3-velo-tcp | 668214 | 0 | 0.000000% | 0 | 30538 |
| r4-velo-tcp | 674203 | 0 | 0.000000% | 0 | 28392 |
| r1-velo-rdma | 668310 | 0 | 0.000000% | 0 | 30244 |
| r2-velo-rdma | 665462 | 1 | 0.000150% | 0 | 30822 |
| r3-velo-rdma | 672621 | 1 | 0.000149% | 0 | 35858 |
| r4-velo-rdma | 678135 | 2 | 0.000295% | 0 | 29330 |

The Ethernet and RDMA fabrics differ. Hardware counters cover all traffic on the measured frontend ports. Totals sum the Ethernet interface and the two selected InfiniBand ports. Host packet aggregation and RDMA completions are not wire packets. Latency percentiles describe successful exported requests; requests cancelled at the drain deadline are counted separately and limit tail comparisons. Packet changes are reported separately from performance qualification. Defaults are unchanged.
