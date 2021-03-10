#!/bin/bash
echo "Start program ..."

echo " dataset ..."
python -i assignment1.py -p 'data/SPX_2012_now.csv' -c 5 -e 'yes' -s '1'
# python assignment1.py -p 'data/Nile.dat' -c 0 -e 'yes' -s '1871'
# python assignment1.py -p 'data/Nile.dat' -c 0 -e 'no' -m '[[21, 40], [61, 80]]' -s '1871'
# python assignment1.py -p 'data/Nile.dat' -c 0 -e 'no' -f '30' -s '1871'
# python assignment1.py -p 'data/Nile.dat' -c 0 -e 'no' -m '[[21, 40], [61, 80]]' -f '30' -s '1871'
