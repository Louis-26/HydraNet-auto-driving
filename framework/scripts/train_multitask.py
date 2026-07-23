from __future__ import annotations

import argparse

from framework.scripts.object_detection.train_detection import build_parser as build_detection_parser, run_training


def build_parser():
    parser = build_detection_parser()
    parser.description = "Multi-task training entry point. In this cleaned branch, the active path is YOLO detection head fine-tuning."
    parser.add_argument("--head", choices=["yolo"], default="yolo", help="Only the YOLO head is supported here.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
