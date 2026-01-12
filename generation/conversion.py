"""Conversion utilities for transforming generated tokens to PCAP format.

This module provides functionality to convert the raw text output from
NetSSM's generate.py script into valid PCAP files that can be analyzed
with standard network analysis tools.

Usage:
    uv run generation/conversion.py <input_txt> <output_pcap> [--ts_sample_gt <gt_pcap>]
"""

import argparse
import os
import random

import scapy.all as scapy


def modify_timestamps(gen_packets, gt_pcap_path, output_path):
    """Modify packet timestamps by sampling from a ground truth PCAP.

    Args:
        gen_packets: List of generated Scapy packets.
        gt_pcap_path: Path to ground truth PCAP for timestamp distribution.
        output_path: Path to write the output PCAP with modified timestamps.
    """
    gt_pkts = scapy.rdpcap(gt_pcap_path)

    # Calculate time differences for ground truth packets
    time_diffs_b = [
        gt_pkts[i].time - gt_pkts[i - 1].time for i in range(1, len(gt_pkts))
    ]

    # Use the time differences of grund truth packets as a statistical sample to draw from
    sampled_time_diffs = random.choices(
        time_diffs_b, k=len(gen_packets) - 1
    )  # -1 because we've already set the first timestamp to 0

    new_timestamps = [
        gen_packets[0].time
    ]  # Start with the original timestamp of the first packet
    for diff in sampled_time_diffs:
        new_timestamps.append(new_timestamps[-1] + diff)

    for i, packet in enumerate(gen_packets):
        packet.time = new_timestamps[i]

    scapy.wrpcap(output_path, gen_packets)


def parse_pcap_string(pcap_string):
    """
    Parses a string-based representation of packets and returns a list of tuples,
    where each tuple contains the interarrival time and the corresponding packet bytes.
    """
    packet_data_strings = pcap_string.split("<|pkt|>")
    packet_data_strings = packet_data_strings[
        1:-1
    ]  # Remove the first label token, and trailing byte

    packets = []
    for packet_str in packet_data_strings:
        values = packet_str.strip().split()
        if not values:
            continue

        try:
            byte_values = values[0:]
            packet_bytes = bytes([int(b) for b in byte_values])
            packets.append(packet_bytes)
        except ValueError:
            continue  # Skip any invalid packets

    return packets


def convert_to_pcap(input_file, output_file, gt_pcap):
    """Convert a text file of generated tokens to a PCAP file.

    Args:
        input_file: Path to the generated text file.
        output_file: Path for the output PCAP file.
        gt_pcap: Optional path to ground truth PCAP for timestamp sampling.
    """
    # Read the raw data from the input file
    with open(input_file, "r") as file:
        pcap_string = file.read()

    if "Could not parse token mapping" in pcap_string:
        return

    packets = parse_pcap_string(pcap_string)

    scapy_packets = []

    for packet_bytes in packets:
        try:
            scapy_packet = scapy.Ether(packet_bytes)
        except Exception as e:
            print(e)
            print(packet_bytes)
            continue
        scapy_packets.append(scapy_packet)

    if os.path.isfile(gt_pcap):
        modify_timestamps(scapy_packets, gt_pcap, output_file)
    else:
        scapy.wrpcap(output_file, scapy_packets)

    print(f"PCAP saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert string-based packet data into a PCAP file."
    )
    parser.add_argument(
        "input_file", help="The file containing the string-based packet data."
    )
    parser.add_argument(
        "--ts_sample_gt",
        help="Path to ground truth PCAP. If provided, will sample timestamps from the distribution of this capture.",
        default="",
    )
    parser.add_argument("output_file", help="The name of the output PCAP file.")

    args = parser.parse_args()

    convert_to_pcap(args.input_file, args.output_file, args.ts_sample_gt)
