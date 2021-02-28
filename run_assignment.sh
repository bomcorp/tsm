#!/bin/bash
echo "Start program ..."

echo "Nile dataset ..."
python assignment1.py -p 'data/Nile.dat' -c 0 -e 'no' -s '1871'
python assignment1.py -p 'data/Nile.dat' -c 0 -e 'yes' -s '1871'
python assignment1.py -p 'data/Nile.dat' -c 0 -e 'no' -m '[[21, 40], [61, 80]]' -s '1871'
python assignment1.py -p 'data/Nile.dat' -c 0 -e 'no' -f '30' -s '1871'
python assignment1.py -p 'data/Nile.dat' -c 0 -e 'no' -m '[[21, 40], [61, 80]]' -f '30' -s '1871'


echo "gdp dataset ..."
python assignment1.py -p 'data/netherlands-gdp-growth-rate.csv' -c 1 -e 'yes' -s '1961'
python -i assignment1.py -p 'data/netherlands-gdp-growth-rate.csv' -c 1 -e 'yes' -m '[[10, 20]]' -s '1961'
python -i assignment1.py -p 'data/netherlands-gdp-growth-rate.csv' -c 1 -e 'yes' -f '30' -s '1961'

echo "Done ..."