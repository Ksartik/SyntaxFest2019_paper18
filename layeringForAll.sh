#!/bin/bash
langs=($(ls data_in_txt/))

for lang in ${langs[*]}
do
    python layering.py "${lang%.txt}";
done