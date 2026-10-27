/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

interface Ipv4Range {
  readonly octets: readonly number[];
  readonly prefix: number;
  /** The written address as a number. */
  readonly address: number;
}

function parseIpv4Cidr(value: string): Ipv4Range | undefined {
  const match =
    /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\/(\d{1,2})$/.exec(
      value,
    );
  if (match === null) {
    return undefined;
  }
  const octets = match.slice(1, 5).map(Number);
  const prefix = Number(match[5]);
  if (octets.some((octet) => octet > 255) || prefix > 32) {
    return undefined;
  }
  const address = octets.reduce((sum, octet) => sum * 256 + octet, 0);
  return { octets, prefix, address };
}

/** True for an IPv4 range such as `10.42.0.0/16`. */
export function isValidIpv4Cidr(value: string): boolean {
  return parseIpv4Cidr(value) !== undefined;
}

/**
 * True for a private, network-aligned range from /16 to /24 that is written
 * without leading zeros, such as `10.42.0.0/16`.
 */
export function isValidVpcCidr(value: string): boolean {
  const range = parseIpv4Cidr(value);
  if (range === undefined || range.prefix < 16 || range.prefix > 24) {
    return false;
  }
  const [first, second] = range.octets;
  const isPrivate =
    first === 10 ||
    (first === 172 && second >= 16 && second <= 31) ||
    (first === 192 && second === 168);
  return (
    isPrivate &&
    range.address % 2 ** (32 - range.prefix) === 0 &&
    `${range.octets.join(".")}/${range.prefix}` === value
  );
}

/** True when two IPv4 ranges share an address. */
export function cidrsOverlap(a: string, b: string): boolean {
  const bounds = (value: string): [number, number] => {
    const range = parseIpv4Cidr(value);
    if (range === undefined) {
      throw new Error(`${value} is not an IPv4 CIDR range.`);
    }
    const size = 2 ** (32 - range.prefix);
    const start = range.address - (range.address % size);
    return [start, start + size];
  };
  const [startA, endA] = bounds(a);
  const [startB, endB] = bounds(b);
  return startA < endB && startB < endA;
}
