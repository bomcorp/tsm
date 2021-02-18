#!/bin/bash
echo "Start program ..."

echo "Nile dataset ..."
#python assignment1.py -p 'data/Nile.dat' -c 0 -e 'no' -m '[[21, 40], [61, 80]]' -f '30'
python assignment1.py -p 'data/Nile.dat' -c 0 -e 'no'

echo "gdp dataset ..."
python -i assignment1.py -p 'data/netherlands-gdp-growth-rate.csv' -c 1 -e 'yes'


echo "Done ..."