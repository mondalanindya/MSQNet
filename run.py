#!/usr/bin/env python3
"""
MSQNet: Multimodal Semantic Query Network for Action Recognition
Top-level entrypoint for training and evaluating MSQNet models.
"""
import os
import sys

# Ensure multi-label-action-main is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(BASE_DIR, 'multi-label-action-main')
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from main import parser, main

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
