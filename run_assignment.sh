#!/bin/bash
echo "Start program ..."

echo "Nile dataset ..."
python assignment1.py -p 'data/Nile.dat' -c 0 -e 'no'
python assignment1.py -p 'data/Nile.dat' -c 0 -e 'yes'
python assignment1.py -p 'data/Nile.dat' -c 0 -e 'no' -m '[[21, 40], [61, 80]]'
python assignment1.py -p 'data/Nile.dat' -c 0 -e 'no' -f '30'
python assignment1.py -p 'data/Nile.dat' -c 0 -e 'no' -m '[[21, 40], [61, 80]]' -f '30'


echo "gdp dataset ..."
python assignment1.py -p 'data/netherlands-gdp-growth-rate.csv' -c 1 -e 'yes'
python assignment1.py -p 'data/netherlands-gdp-growth-rate.csv' -c 1 -e 'yes' -m '[[21, 40]]'
python -i assignment1.py -p 'data/netherlands-gdp-growth-rate.csv' -c 1 -e 'yes' -f '30'

echo "Done ..."