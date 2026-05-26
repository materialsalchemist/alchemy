#!/bin/bash

# Build script for SCSS
# Requirements: sass (Dart Sass)

# Ensure we are in the project root or the app directory
# This script assumes it is run from the project root.

INPUT="app/style.scss"
OUTPUT="app/static/css/style.css"

echo "Building CSS using npx sass..."

npx sass "$INPUT" "$OUTPUT" --no-source-map

if [ $? -eq 0 ]; then
    echo "Successfully built $OUTPUT"
else
    echo "Error: Sass compilation failed."
    exit 1
fi
