#!/bin/bash

find .. -type d -name "__pycache__" -exec rm -rf {} +;
find .. -type f -name "*.pyc" -delete;