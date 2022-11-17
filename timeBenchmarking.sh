#!/usr/bin/env bash

echo "Running make"
(make clean > /dev/null) && (make > /dev/null)

outputFile=timeBenchmarks.txt

if [ -f $outputFile ]
then
    rm $outputFile
fi

echo "Running "

time ./main-app >> $outputFile
echo >> $outputFile

echo "Done"
