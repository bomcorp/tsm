#!/bin/bash
echo "Start program ..."

echo "sv dataset ..."
python -i assignment2.py -p 'data/sv.dat' -c 0 -e 'no' -s '1'
#python -i assignment2.py -p 'data/sv.dat' -c 0 -e 'yes' -s '1871'
#python assignment2.py -p 'data/Nile.dat' -c 0 -e 'no' -m '[[21, 40], [61, 80]]' -f '30' -s '1871'