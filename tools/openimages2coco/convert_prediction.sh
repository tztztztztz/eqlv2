#!/usr/bin/env bash

result_file=$1
python convert_predictions.py -p ${result_file} --subset validation